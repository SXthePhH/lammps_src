// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Langevin-NH: drop-in replacement for fix_nh that replaces the
   Nosé-Hoover chain (NHC) thermostat and NHC barostat thermostat with
   Langevin stochastic dynamics, while keeping the EXACT same
   integration order skeleton as fix_nh.

   Integration order (initial_integrate):
     1. [was nhc_press_integrate]  → langevin_press()     (O-step on omega_dot)
     2. [was nhc_temp_integrate]   → langevin_temp()      (O-step on v)
     3. recompute T, P
     4. nh_omega_dot()                                    (B half-step on omega)
     5. nh_v_press()                                      (B half-step v scale)
     6. nve_v()                                           (B half-step force kick)
     7. remap()                                           (A half-step box)
     8. nve_x()                                           (A full-step positions)
     9. remap() + kspace setup                            (A half-step box)

   Integration order (final_integrate):
     1. nve_v()                                           (B half-step force kick)
     2. nh_v_press()                                      (B half-step v scale)
     3. recompute T, P
     4. nh_omega_dot()                                    (B half-step on omega)
     5. [was nhc_temp_integrate]   → langevin_temp()      (O-step on v)
     6. [was nhc_press_integrate]  → langevin_press()     (O-step on omega_dot)

   New keyword syntax (replacing tchain/pchain/tloop/ploop):
     temp  Tstart Tstop damp_t
     iso / aniso / tri / x / y / z / yz / xz / xy:
           Pstart Pstop damp_p    (damp_p replaces the NH period arg)
     seed  S                      (RNG seed, default 12345)
   All other fix_nh keywords kept as-is.
------------------------------------------------------------------------- */

#include "fix_nh_langevin.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "force.h"
#include "group.h"
#include "irregular.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double DELTAFLIP = 0.1;
static constexpr double TILTMAX   = 1.5;
static constexpr double EPSILON   = 1.0e-6;

enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};
enum{NOBIAS,BIAS};

static bool nh_langevin_debug_enabled()
{
  static int cached = -1;
  if (cached < 0) cached = (std::getenv("NH_LANGEVIN_DEBUG") != nullptr) ? 1 : 0;
  return cached != 0;
}

static void nh_langevin_debug(const char *phase, const char *style, int narg, char **arg, int iarg = -1)
{
  if (!nh_langevin_debug_enabled()) return;
  if (!style) style = "<null>";
  fprintf(stderr, "[nh/langevin debug] phase=%s style=%s narg=%d", phase, style, narg);
  if (iarg >= 0 && iarg < narg) fprintf(stderr, " iarg=%d token=%s", iarg, arg[iarg] ? arg[iarg] : "<null>");
  fprintf(stderr, "\n");
}

/* ======================================================================
   Constructor
   ====================================================================== */

FixNHLangevin::FixNHLangevin(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg),
    random(nullptr),
    id_dilate(nullptr), irregular(nullptr), step_respa(nullptr),
    id_temp(nullptr), id_press(nullptr)
{
  const char *fixname = style ? style : "nh/langevin";

  nh_langevin_debug("ctor-enter", style, narg, arg);

  if (narg < 4) utils::missing_cmd_args(FLERR, std::string("fix ") + fixname, error);

  restart_global        = 1;
  dynamic_group_allow   = 1;
  thermo_modify_colname = 1;
  time_integrate        = 1;
  scalar_flag           = 1;
  vector_flag           = 1;
  global_freq           = 1;
  extscalar             = 1;
  extvector             = 0;
  ecouple_flag          = 1;

  // ---- defaults (identical to fix_nh) ----
  pcouple         = NONE;
  drag            = 0.0;
  allremap        = 1;
  id_dilate       = nullptr;
  mtk_flag        = 1;
  deviatoric_flag = 0;
  nreset_h0       = 0;
  flipflag        = 1;
  dipole_flag     = 0;
  dlm_flag        = 0;
  p_temp_flag     = 0;
  tcomputeflag    = 0;
  pcomputeflag    = 0;

  // subclass-compat stubs
  mtchain_default_flag = 1;
  eta_mass_flag        = 0;
  omega_mass_flag      = 0;
  etap_mass_flag       = 0;
  factor_eta           = 1.0;
  tdrag_factor         = 1.0;

  // ---- Langevin defaults ----
  damp_t   = 0.0;
  damp_p   = 0.0;
  lan_seed = 12345;
  gamma_t  = gamma_p  = 0.0;
  lan_c1_t = lan_c1_p = 0.0;
  lan_c2_t = lan_c2_p = 0.0;

  dimension = domain->dimension;

  scaleyz = scalexz = scalexy = 0;
  if (domain->yperiodic && domain->xy != 0.0) scalexy = 1;
  if (domain->zperiodic && dimension == 3) {
    if (domain->yz != 0.0) scaleyz = 1;
    if (domain->xz != 0.0) scalexz = 1;
  }

  fixedpoint[0] = 0.5*(domain->boxlo[0]+domain->boxhi[0]);
  fixedpoint[1] = 0.5*(domain->boxlo[1]+domain->boxhi[1]);
  fixedpoint[2] = 0.5*(domain->boxlo[2]+domain->boxhi[2]);

  tstat_flag = 0;
  t_freq     = 0.0;

  double p_period[6];
  for (int i = 0; i < 6; i++) {
    p_start[i] = p_stop[i] = p_period[i] = p_target[i] = 0.0;
    p_flag[i]  = 0;
  }

  // ---- parse keywords ----
  // "temp"  : Tstart Tstop damp_t          (3 args, period→damping time)
  // pressure keywords: Pstart Pstop damp_p (3 args, period→damping time)
  // "seed"  : S

  int iarg = 3;
  while (iarg < narg) {
    nh_langevin_debug("parse-arg", style, narg, arg, iarg);

    if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} temp", style), error);
      tstat_flag = 1;
      t_start  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      t_target = t_start;
      t_stop   = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      damp_t   = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      if (t_start <= 0.0 || t_stop <= 0.0)
        error->all(FLERR, "Target temperature for fix {} cannot be 0.0", style);
      if (damp_t <= 0.0)
        error->all(FLERR, "Langevin thermostat damp_t must be > 0", style);
      iarg += 4;

    } else if (strcmp(arg[iarg],"iso") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} iso", style), error);
      pcouple = XYZ;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0]  = p_stop[1]  = p_stop[2]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (dimension == 2) { p_start[2] = p_stop[2] = 0.0; p_flag[2] = 0; }
      iarg += 4;

    } else if (strcmp(arg[iarg],"aniso") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} aniso", style), error);
      pcouple = NONE;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0]  = p_stop[1]  = p_stop[2]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (dimension == 2) { p_start[2] = p_stop[2] = 0.0; p_flag[2] = 0; }
      iarg += 4;

    } else if (strcmp(arg[iarg],"tri") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} tri", style), error);
      pcouple = NONE;
      scalexy = scalexz = scaleyz = 0;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0]  = p_stop[1]  = p_stop[2]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      for (int i = 0; i < 3; i++) p_flag[i] = 1;
      p_start[3] = p_start[4] = p_start[5] = 0.0;
      p_stop[3]  = p_stop[4]  = p_stop[5]  = 0.0;
      p_flag[3]  = p_flag[4]  = p_flag[5]  = 1;
      if (dimension == 2) {
        p_start[2] = p_stop[2] = 0.0; p_flag[2] = 0;
        p_start[3] = p_stop[3] = 0.0; p_flag[3] = 0;
        p_start[4] = p_stop[4] = 0.0; p_flag[4] = 0;
      }
      iarg += 4;

    } else if (strcmp(arg[iarg],"x") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} x", style), error);
      p_start[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p     = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0]  = 1; deviatoric_flag = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg],"y") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} y", style), error);
      p_start[1] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[1]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p     = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[1]  = 1; deviatoric_flag = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg],"z") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} z", style), error);
      if (dimension == 2)
        error->all(FLERR,"Invalid fix {} command for a 2d simulation", style);
      p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[2]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p     = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[2]  = 1; deviatoric_flag = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg],"yz") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} yz", style), error);
      if (dimension == 2)
        error->all(FLERR,"Invalid fix {} command for a 2d simulation", style);
      p_start[3] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[3]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p     = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[3]  = 1; deviatoric_flag = 1; scaleyz = 0;
      iarg += 4;

    } else if (strcmp(arg[iarg],"xz") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} xz", style), error);
      if (dimension == 2)
        error->all(FLERR,"Invalid fix {} command for a 2d simulation", style);
      p_start[4] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[4]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p     = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[4]  = 1; deviatoric_flag = 1; scalexz = 0;
      iarg += 4;

    } else if (strcmp(arg[iarg],"xy") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} xy", style), error);
      p_start[5] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[5]  = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      damp_p     = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[5]  = 1; deviatoric_flag = 1; scalexy = 0;
      iarg += 4;

    } else if (strcmp(arg[iarg],"couple") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} couple", style), error);
      if      (strcmp(arg[iarg+1],"xyz")  == 0) pcouple = XYZ;
      else if (strcmp(arg[iarg+1],"xy")   == 0) pcouple = XY;
      else if (strcmp(arg[iarg+1],"yz")   == 0) pcouple = YZ;
      else if (strcmp(arg[iarg+1],"xz")   == 0) pcouple = XZ;
      else if (strcmp(arg[iarg+1],"none") == 0) pcouple = NONE;
      else error->all(FLERR,"Illegal fix {} couple option: {}", style, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} seed", style), error);
      lan_seed = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"drag") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} drag", style), error);
      drag = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (drag < 0.0) error->all(FLERR,"Invalid fix {} drag argument: {}", style, drag);
      iarg += 2;

    } else if (strcmp(arg[iarg],"ptemp") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} ptemp", style), error);
      p_temp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_temp_flag = 1;
      if (p_temp <= 0.0)
        error->all(FLERR,"Invalid fix {} ptemp argument: {}", style, p_temp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"dilate") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} dilate", style), error);
      if (strcmp(arg[iarg+1],"all") == 0) allremap = 1;
      else {
        allremap = 0;
        delete[] id_dilate;
        id_dilate = utils::strdup(arg[iarg+1]);
        int idilate = group->find(id_dilate);
        if (idilate < 0)
          error->all(FLERR,"Fix {} dilate group ID {} does not exist", style, id_dilate);
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"nreset") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} nreset", style), error);
      nreset_h0 = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (nreset_h0 < 0)
        error->all(FLERR,"Invalid fix {} nreset argument: {}", style, nreset_h0);
      iarg += 2;

    } else if (strcmp(arg[iarg],"scalexy") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} scalexy", style), error);
      scalexy = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"scalexz") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} scalexz", style), error);
      scalexz = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"scaleyz") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} scaleyz", style), error);
      scaleyz = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"flip") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} flip", style), error);
      flipflag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} update", style), error);
      if      (strcmp(arg[iarg+1],"dipole")     == 0) dipole_flag = 1;
      else if (strcmp(arg[iarg+1],"dipole/dlm") == 0) { dipole_flag = 1; dlm_flag = 1; }
      else error->all(FLERR,"Invalid fix {} update argument: {}", style, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg],"fixedpoint") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} fixedpoint", style), error);
      fixedpoint[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      fixedpoint[1] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      fixedpoint[2] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      iarg += 4;

    // ---- NHC-specific keywords: warn and skip for script compatibility ----
    } else if (strcmp(arg[iarg],"tchain") == 0 ||
               strcmp(arg[iarg],"pchain") == 0 ||
               strcmp(arg[iarg],"mtk")    == 0 ||
               strcmp(arg[iarg],"tloop")  == 0 ||
               strcmp(arg[iarg],"ploop")  == 0) {
      if (comm->me == 0)
        error->warning(FLERR,"fix {} keyword '{}' is not used in Langevin mode",
                       style, arg[iarg]);
      iarg += 2;

    // ---- pass-through keywords for subclasses ----
    } else if (strcmp(arg[iarg],"disc")       == 0) { iarg++;
    } else if (strcmp(arg[iarg],"erate")      == 0) { iarg += 3;
    } else if (strcmp(arg[iarg],"strain")     == 0) { iarg += 3;
    } else if (strcmp(arg[iarg],"ext")        == 0) { iarg += 2;
    } else if (strcmp(arg[iarg],"psllod")     == 0) { iarg += 2;
    } else if (strcmp(arg[iarg],"peculiar")   == 0) { iarg += 2;
    } else if (strcmp(arg[iarg],"kick")       == 0) { iarg += 2;
    } else if (strcmp(arg[iarg],"integrator") == 0) { iarg += 2;
    } else {
      error->all(FLERR,"Unknown fix {} keyword: {}", style, arg[iarg]);
    }
  }

  nh_langevin_debug("ctor-exit", style, narg, arg);

  // ---- same geometric/coupling error checks as fix_nh ----

  if (dimension == 2 && (p_flag[2] || p_flag[3] || p_flag[4]))
    error->all(FLERR,"Invalid fix {} command for a 2d simulation", style);
  if (dimension == 2 && (pcouple == YZ || pcouple == XZ))
    error->all(FLERR,"Invalid fix {} command for a 2d simulation", style);
  if (dimension == 2 && (scalexz == 1 || scaleyz == 1))
    error->all(FLERR,"Invalid fix {} command for a 2d simulation", style);

  if (pcouple == XYZ && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix {} command pressure settings", style);
  if (pcouple == XYZ && dimension == 3 && p_flag[2] == 0)
    error->all(FLERR,"Invalid fix {} command pressure settings", style);
  if (pcouple == XY  && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix {} command pressure settings", style);
  if (pcouple == YZ  && (p_flag[1] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix {} command pressure settings", style);
  if (pcouple == XZ  && (p_flag[0] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix {} command pressure settings", style);

  if (p_flag[0] && domain->xperiodic == 0)
    error->all(FLERR,"Cannot use fix {} on a non-periodic x dimension", style);
  if (p_flag[1] && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix {} on a non-periodic y dimension", style);
  if (p_flag[2] && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix {} on a non-periodic z dimension", style);
  if (p_flag[3] && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix {} on a 2nd non-periodic dimension", style);
  if (p_flag[4] && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix {} on a 2nd non-periodic dimension", style);
  if (p_flag[5] && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix {} on a 2nd non-periodic dimension", style);

  if (scaleyz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix {} with yz scaling when z is non-periodic dimension",style);
  if (scalexz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix {} with xz scaling when z is non-periodic dimension",style);
  if (scalexy == 1 && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix {} with xy scaling when y is non-periodic dimension",style);

  if (p_flag[3] && scaleyz == 1)
    error->all(FLERR,"Cannot use fix {} with both yz dynamics and yz scaling",style);
  if (p_flag[4] && scalexz == 1)
    error->all(FLERR,"Cannot use fix {} with both xz dynamics and xz scaling",style);
  if (p_flag[5] && scalexy == 1)
    error->all(FLERR,"Cannot use fix {} with both xy dynamics and xy scaling",style);

  if (!domain->triclinic && (p_flag[3] || p_flag[4] || p_flag[5]))
    error->all(FLERR,"Can not specify Pxy/Pxz/Pyz in fix {} with non-triclinic box",style);

  if (dipole_flag) {
    if (strstr(style, "/sphere")) {
      if (!atom->omega_flag)
        error->all(FLERR,"Using update dipole flag requires atom attribute omega");
      if (!atom->radius_flag)
        error->all(FLERR,"Using update dipole flag requires atom attribute radius");
      if (!atom->mu_flag)
        error->all(FLERR,"Using update dipole flag requires atom attribute mu");
    } else {
      error->all(FLERR,"Must use a '/sphere' fix style for updating dipoles");
    }
  }

  if (tstat_flag && damp_t <= 0.0)
    error->all(FLERR,"Fix {} thermostat damp_t must be > 0", style);
  if (tstat_flag && p_temp_flag)
    error->all(FLERR,"Thermostat in fix {} is incompatible with ptemp command", style);

  // ---- pstat flags and box-change (identical to fix_nh) ----
  pre_exchange_flag = 0;
  pstat_flag = 0;
  pstyle = ISO;

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) pstat_flag = 1;

  if (pstat_flag) {
    if (p_flag[0]) box_change |= BOX_CHANGE_X;
    if (p_flag[1]) box_change |= BOX_CHANGE_Y;
    if (p_flag[2]) box_change |= BOX_CHANGE_Z;
    if (p_flag[3]) box_change |= BOX_CHANGE_YZ;
    if (p_flag[4]) box_change |= BOX_CHANGE_XZ;
    if (p_flag[5]) box_change |= BOX_CHANGE_XY;
    no_change_box = 1;
    if (allremap == 0) restart_pbc = 1;

    if (p_flag[3] || p_flag[4] || p_flag[5]) pstyle = TRICLINIC;
    else if (pcouple == XYZ || (dimension == 2 && pcouple == XY)) pstyle = ISO;
    else pstyle = ANISO;

    if (flipflag && (p_flag[3] || p_flag[4] || p_flag[5]))
      pre_exchange_flag = pre_exchange_migrate = 1;
    if (flipflag && (domain->yz != 0.0 || domain->xz != 0.0 || domain->xy != 0.0))
      pre_exchange_flag = pre_exchange_migrate = 1;

    if (damp_p <= 0.0)
      error->all(FLERR,"Fix {} barostat damp_p must be > 0", style);
  }

  // Create internal compute objects the same way fix_npt/fix_nph do.
  // init() expects id_temp to exist even without an explicit fix_modify temp.
  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp", id_temp));
  tcomputeflag = 1;

  if (pstat_flag) {
    id_press = utils::strdup(std::string(id) + "_press");
    modify->add_compute(fmt::format("{} all pressure {}", id_press, id_temp));
    pcomputeflag = 1;
  }

  // ---- omega arrays ----
  for (int i = 0; i < 6; i++)
    omega[i] = omega_dot[i] = omega_mass[i] = 0.0;

  // size_vector: omega + omega_dot only (no NHC eta terms)
  size_vector = 0;
  if (pstat_flag) {
    if      (pstyle == ISO)       size_vector += 2*2*1;
    else if (pstyle == ANISO)     size_vector += 2*2*3;
    else if (pstyle == TRICLINIC) size_vector += 2*2*6;
    if (deviatoric_flag) size_vector += 1;
  }

  if (pre_exchange_flag) irregular = new Irregular(lmp);
  else irregular = nullptr;

  vol0 = t0 = 0.0;

  random = new RanMars(lmp, lan_seed + comm->me);

  // p_freq initialised to 0; set properly in setup() once damp_p is final
  for (int i = 0; i < 6; i++) p_freq[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixNHLangevin::~FixNHLangevin()
{
  if (copymode) return;

  delete random;
  delete[] id_dilate;
  delete irregular;

  if (tcomputeflag) modify->delete_compute(id_temp);
  delete[] id_temp;

  if (pstat_flag) {
    if (pcomputeflag) modify->delete_compute(id_press);
    delete[] id_press;
  }
}

/* ---------------------------------------------------------------------- */

int FixNHLangevin::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  mask |= PRE_FORCE_RESPA;
  if (pre_exchange_flag) mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::init()
{
  nh_langevin_debug("init-enter", style, 0, nullptr);

  if (allremap == 0)
    dilate_group_bit = group->get_bitmask_by_id(FLERR, id_dilate,
                                                 fmt::format("fix {}", style));

  if (pstat_flag)
    for (const auto &ifix : modify->get_fix_by_style("^deform")) {
      auto *deform = dynamic_cast<FixDeform *>(ifix);
      if (deform) {
        int *dimflag = deform->dimflag;
        if ((p_flag[0] && dimflag[0]) || (p_flag[1] && dimflag[1]) ||
            (p_flag[2] && dimflag[2]) || (p_flag[3] && dimflag[3]) ||
            (p_flag[4] && dimflag[4]) || (p_flag[5] && dimflag[5]))
          error->all(FLERR,"Cannot use fix {} and fix deform on "
                     "same component of stress tensor", style);
      }
    }

  temperature = modify->get_compute_by_id(id_temp);
  if (!temperature)
    error->all(FLERR,"Temperature compute ID {} for fix {} does not exist", id_temp, style);
  if (temperature->tempflag == 0)
    error->all(FLERR,"Compute ID {} for fix {} does not compute a temperature", id_temp, style);
  if (temperature->tempbias) which = BIAS;
  else which = NOBIAS;

  if (pstat_flag) {
    pressure = modify->get_compute_by_id(id_press);
    if (!pressure)
      error->all(FLERR,"Pressure compute ID {} for fix {} does not exist", id_press, style);
    if (pressure->pressflag == 0)
      error->all(FLERR,"Compute ID {} for fix {} does not compute pressure", id_press, style);
  }

  // ---- timestep quantities (identical to fix_nh) ----
  dtv    = update->dt;
  dtf    = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4    = 0.25 * update->dt;
  dt8    = 0.125 * update->dt;
  dto    = dthalf;

  boltz  = force->boltz;
  nktv2p = force->nktv2p;

  // ---- Langevin friction / OU coefficients ----
  gamma_t  = 1.0 / damp_t;
  // Half-step O coefficients for thermostat
  lan_c1_t = exp(-gamma_t * dthalf);
  lan_c2_t = sqrt((1.0 - lan_c1_t * lan_c1_t) * boltz * t_target);

  // Drag factor (mirrors fix_nh; drag=0 → pdrag_factor=1)
  if (pstat_flag) {
    gamma_p      = 1.0 / damp_p;
    lan_c1_p     = exp(-gamma_p * dthalf);
    lan_c2_p     = sqrt((1.0 - lan_c1_p * lan_c1_p) * boltz * t_target);
    p_freq_max   = 1.0 / damp_p;
    pdrag_factor = 1.0 - (update->dt * p_freq_max * drag / 1.0);
    // set p_freq[i] = 1/damp_p for omega_mass formula
    for (int i = 0; i < 6; i++) p_freq[i] = p_freq_max;
  }

  if (tstat_flag)
    tdrag_factor = 1.0 - (update->dt * (1.0/damp_t) * drag / 1.0);

  // ---- volume / reference cell ----
  if (pstat_flag) {
    pdim = p_flag[0] + p_flag[1] + p_flag[2];
    if (vol0 == 0.0) {
      if (dimension == 3) vol0 = domain->xprd * domain->yprd * domain->zprd;
      else vol0 = domain->xprd * domain->yprd;
      for (int i = 0; i < 6; i++) h0_inv[i] = domain->h_inv[i];
    }
  }

  if (force->kspace) kspace_flag = 1;
  else kspace_flag = 0;

  if (utils::strmatch(update->integrate_style,"^respa")) {
    auto *respa_ptr = dynamic_cast<Respa *>(update->integrate);
    if (!respa_ptr)
      error->all(FLERR,"Failure to access Respa style {}", update->integrate_style);
    nlevels_respa = respa_ptr->nlevels;
    step_respa    = respa_ptr->step;
    dto = 0.5 * step_respa[0];
  }

  rfix.clear();
  for (const auto &ifix : modify->get_fix_list())
    if (ifix->rigid_flag) rfix.push_back(ifix);

  if (nh_langevin_debug_enabled()) {
    fprintf(stderr,
            "[nh/langevin debug] init-state style=%s tstat=%d pstat=%d t_target=%g t0=%g damp_t=%g damp_p=%g pstyle=%d pcouple=%d pflags=%d%d%d%d%d%d\n",
            style ? style : "<null>", tstat_flag, pstat_flag, t_target, t0, damp_t, damp_p,
            pstyle, pcouple, p_flag[0], p_flag[1], p_flag[2], p_flag[3], p_flag[4], p_flag[5]);
  }
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::setup(int /*vflag*/)
{
  nh_langevin_debug("setup-enter", style, 0, nullptr);

  t_current = temperature->compute_scalar();
  tdof      = temperature->dof;   // tdof is double

  if (tstat_flag && strstr(style,"nphug") == nullptr) {
    compute_temp_target();
  } else if (pstat_flag) {
    if (t0 == 0.0) {
      if (p_temp_flag) {
        t0 = p_temp;
      } else {
        t0 = temperature->compute_scalar();
        if (t0 < EPSILON)
          error->all(FLERR,"Current temperature too close to zero, "
                     "consider using ptemp keyword");
      }
    }
    t_target = t0;
  }

  // Update Langevin OU coefficients using actual t_target
  lan_c1_t = exp(-gamma_t * dthalf);
  lan_c2_t = sqrt((1.0 - lan_c1_t * lan_c1_t) * boltz * t_target);
  if (pstat_flag) {
    lan_c1_p = exp(-gamma_p * dthalf);
    lan_c2_p = sqrt((1.0 - lan_c1_p * lan_c1_p) * boltz * t_target);
  }

  if (pstat_flag) compute_press_target();

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  // ---- omega masses (same formula as fix_nh: W = (N+1)*kT/p_freq^2) ----
  if (pstat_flag) {
    double kt  = boltz * t_target;
    double nkt = (atom->natoms + 1) * kt;
    for (int i = 0; i < 3; i++)
      if (p_flag[i]) omega_mass[i] = nkt / (p_freq[i]*p_freq[i]);
    if (pstyle == TRICLINIC)
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt / (p_freq[i]*p_freq[i]);
  }

  if (nh_langevin_debug_enabled()) {
    fprintf(stderr,
            "[nh/langevin debug] setup-state style=%s t_current=%g tdof=%g t_target=%g ke_target=%g p_target=%g,%g,%g p_freq=%g,%g,%g omega_mass=%g,%g,%g\n",
            style ? style : "<null>", t_current, tdof, t_target, ke_target,
            p_target[0], p_target[1], p_target[2], p_freq[0], p_freq[1], p_freq[2],
            omega_mass[0], omega_mass[1], omega_mass[2]);
  }
}

/* ======================================================================
   1st half of Verlet update — identical call order to fix_nh
   NHC calls replaced by Langevin O-steps
   ====================================================================== */

void FixNHLangevin::initial_integrate(int /*vflag*/)
{
  // [was nhc_press_integrate] — Langevin O-step on barostat
  if (pstat_flag) langevin_press();

  // [was nhc_temp_integrate] — Langevin O-step on particle velocities
  if (tstat_flag) {
    compute_temp_target();
    langevin_temp();
  }

  // recompute T and P to account for velocity change (same as fix_nh)
  if (pstat_flag) {
    if (pstyle == ISO) {
      t_current = temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      t_current = temperature->compute_scalar();
      tdof = temperature->dof;
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  // half-step barostat velocity (B-step)
  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  // half-step force kick (B-step)
  nve_v();

  // half-step box remap (A-step)
  if (pstat_flag) remap();

  // full-step position update (A-step)
  nve_x();

  // second half-step box remap + kspace setup (A-step)
  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

/* ======================================================================
   2nd half of Verlet update — identical call order to fix_nh
   ====================================================================== */

void FixNHLangevin::final_integrate()
{
  nve_v();

  // re-compute temp before nh_v_press() on reneighboring steps with BIAS
  if (which == BIAS && neighbor->ago == 0)
    t_current = temperature->compute_scalar();

  if (pstat_flag) nh_v_press();

  // recompute T and P after velocity rescaling
  t_current = temperature->compute_scalar();
  tdof      = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot();

  // [was nhc_temp_integrate]
  if (tstat_flag) langevin_temp();

  // [was nhc_press_integrate]
  if (pstat_flag) langevin_press();
}

/* ======================================================================
   RESPA — identical to fix_nh, NHC calls → Langevin
   ====================================================================== */

void FixNHLangevin::initial_integrate_respa(int /*vflag*/, int ilevel, int /*iloop*/)
{
  dtv    = step_respa[ilevel];
  dtf    = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];

  if (ilevel == nlevels_respa-1) {

    if (pstat_flag) langevin_press();

    if (tstat_flag) {
      compute_temp_target();
      langevin_temp();
    }

    if (pstat_flag) {
      if (pstyle == ISO) {
        t_current = temperature->compute_scalar();
        pressure->compute_scalar();
      } else {
        t_current = temperature->compute_scalar();
        tdof = temperature->dof;
        temperature->compute_vector();
        pressure->compute_vector();
      }
      couple();
      pressure->addstep(update->ntimestep+1);
    }

    if (pstat_flag) {
      compute_press_target();
      nh_omega_dot();
      nh_v_press();
    }

    nve_v();
  } else nve_v();

  if (ilevel == 0) {
    if (pstat_flag) remap();
    nve_x();
    if (pstat_flag) remap();
  }
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::pre_force_respa(int /*vflag*/, int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1 && kspace_flag && pstat_flag)
    force->kspace->setup();
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf    = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];
  if (ilevel == nlevels_respa-1) final_integrate();
  else nve_v();
}

/* ======================================================================
   langevin_temp — Langevin O-step on particle velocities
   Replaces nhc_temp_integrate.

   Ornstein-Uhlenbeck half-step (BAOAB):
     v_i ← c1_t * v_i + c2_t * (1/sqrt(m_i)) * xi_i
   where c1 = exp(-gamma_t * dthalf), c2 = sqrt((1-c1^2)*kT).
   The 1/sqrt(m) factor is absorbed into each atom's kick magnitude.
   ====================================================================== */

void FixNHLangevin::langevin_temp()
{
  double **v   = atom->v;
  double *mass = atom->mass;
  double *rmass= atom->rmass;
  double mvv2e = force->mvv2e;
  int *mask    = atom->mask;
  int nlocal   = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Refresh coefficients for current t_target (handles temperature ramps)
  lan_c1_t = exp(-gamma_t * dthalf);
  lan_c2_t = sqrt((1.0 - lan_c1_t * lan_c1_t) * boltz * t_target);

  if (which == NOBIAS) {
    if (rmass) {
      for (int i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        double inv_sqrt_m = 1.0 / sqrt(rmass[i] * mvv2e);
        v[i][0] = lan_c1_t * v[i][0] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][1] = lan_c1_t * v[i][1] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][2] = lan_c1_t * v[i][2] + lan_c2_t * random->gaussian() * inv_sqrt_m;
      }
    } else {
      for (int i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        double inv_sqrt_m = 1.0 / sqrt(mass[atom->type[i]] * mvv2e);
        v[i][0] = lan_c1_t * v[i][0] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][1] = lan_c1_t * v[i][1] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][2] = lan_c1_t * v[i][2] + lan_c2_t * random->gaussian() * inv_sqrt_m;
      }
    }
  } else {
    // BIAS: remove bias, apply O-step, restore bias
    if (rmass) {
      for (int i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        temperature->remove_bias(i, v[i]);
        double inv_sqrt_m = 1.0 / sqrt(rmass[i] * mvv2e);
        v[i][0] = lan_c1_t * v[i][0] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][1] = lan_c1_t * v[i][1] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][2] = lan_c1_t * v[i][2] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        temperature->restore_bias(i, v[i]);
      }
    } else {
      for (int i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        temperature->remove_bias(i, v[i]);
        double inv_sqrt_m = 1.0 / sqrt(mass[atom->type[i]] * mvv2e);
        v[i][0] = lan_c1_t * v[i][0] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][1] = lan_c1_t * v[i][1] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        v[i][2] = lan_c1_t * v[i][2] + lan_c2_t * random->gaussian() * inv_sqrt_m;
        temperature->restore_bias(i, v[i]);
      }
    }
  }
  // Note: t_current is NOT updated here.  The next explicit
  // temperature->compute_scalar() call in initial/final_integrate
  // provides the authoritative value used by nh_omega_dot().
}

/* ======================================================================
   langevin_press — Langevin O-step on barostat velocities omega_dot[6]
   Replaces nhc_press_integrate.

   omega_dot_i ← c1_p * omega_dot_i + kick_i
   kick_i = c2_p / sqrt(W_i) * xi_i

   Generated on rank 0 and broadcast so all ranks apply the same kick,
   preserving reproducibility regardless of decomposition.
   Coupling modes share one random draw (same convention as fix_langevin_npt).
   ====================================================================== */

void FixNHLangevin::langevin_press()
{
  // Refresh coefficients for current t_target
  lan_c1_p = exp(-gamma_p * dthalf);
  lan_c2_p = sqrt((1.0 - lan_c1_p * lan_c1_p) * boltz * t_target);

  double kicks[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  if (comm->me == 0) {
    if (pcouple == XYZ) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[1] = kicks[2] = kick;
    } else if (pcouple == XY) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[1] = kick;
      if (p_flag[2]) kicks[2] = lan_c2_p * random->gaussian() / sqrt(omega_mass[2]);
    } else if (pcouple == YZ) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[1]);
      kicks[1] = kicks[2] = kick;
      if (p_flag[0]) kicks[0] = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
    } else if (pcouple == XZ) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[2] = kick;
      if (p_flag[1]) kicks[1] = lan_c2_p * random->gaussian() / sqrt(omega_mass[1]);
    } else {
      for (int i = 0; i < 6; i++)
        if (p_flag[i]) kicks[i] = lan_c2_p * random->gaussian() / sqrt(omega_mass[i]);
    }
  }

  MPI_Bcast(kicks, 6, MPI_DOUBLE, 0, world);

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) omega_dot[i] = lan_c1_p * omega_dot[i] + kicks[i];
}

/* ======================================================================
   Everything below is IDENTICAL to fix_nh.cpp
   ====================================================================== */

void FixNHLangevin::couple()
{
  double *tensor = pressure->vector;

  if (pstyle == ISO)
    p_current[0] = p_current[1] = p_current[2] = pressure->scalar;
  else if (pcouple == XYZ) {
    double ave = 1.0/3.0 * (tensor[0] + tensor[1] + tensor[2]);
    p_current[0] = p_current[1] = p_current[2] = ave;
  } else if (pcouple == XY) {
    double ave = 0.5 * (tensor[0] + tensor[1]);
    p_current[0] = p_current[1] = ave;
    p_current[2] = tensor[2];
  } else if (pcouple == YZ) {
    double ave = 0.5 * (tensor[1] + tensor[2]);
    p_current[1] = p_current[2] = ave;
    p_current[0] = tensor[0];
  } else if (pcouple == XZ) {
    double ave = 0.5 * (tensor[0] + tensor[2]);
    p_current[0] = p_current[2] = ave;
    p_current[1] = tensor[1];
  } else {
    p_current[0] = tensor[0];
    p_current[1] = tensor[1];
    p_current[2] = tensor[2];
  }

  if (!std::isfinite(p_current[0]) || !std::isfinite(p_current[1]) ||
      !std::isfinite(p_current[2]))
    error->all(FLERR,"Non-numeric pressure - simulation unstable" + utils::errorurl(6));

  if (pstyle == TRICLINIC) {
    p_current[3] = tensor[5];
    p_current[4] = tensor[4];
    p_current[5] = tensor[3];
    if (!std::isfinite(p_current[3]) || !std::isfinite(p_current[4]) ||
        !std::isfinite(p_current[5]))
      error->all(FLERR,"Non-numeric pressure - simulation unstable" + utils::errorurl(6));
  }
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::remap()
{
  int i;
  double oldlo, oldhi, expfac;

  double **x = atom->x;
  int *mask  = atom->mask;
  int nlocal = atom->nlocal;
  double *h  = domain->h;

  for (i = 0; i < 6; i++) omega[i] += dto * omega_dot[i];

  if (allremap) domain->x2lamda(nlocal);
  else {
    for (i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit) domain->x2lamda(x[i],x[i]);
  }

  for (auto &ifix : rfix) ifix->deform(0);

  double dto2 = dto/2.0;
  double dto4 = dto/4.0;
  double dto8 = dto/8.0;

  if (pstyle == TRICLINIC) {
    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac; h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]); h[4] *= expfac;
    }
    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac; h[3] += dto2*(omega_dot[3]*h[2]); h[3] *= expfac;
    }
    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac; h[5] += dto2*(omega_dot[5]*h[1]); h[5] *= expfac;
    }
    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac; h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]); h[4] *= expfac;
    }
  }

  if (p_flag[0]) {
    oldlo = domain->boxlo[0]; oldhi = domain->boxhi[0];
    expfac = exp(dto*omega_dot[0]);
    domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
    domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
  }
  if (p_flag[1]) {
    oldlo = domain->boxlo[1]; oldhi = domain->boxhi[1];
    expfac = exp(dto*omega_dot[1]);
    domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
    domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    if (scalexy) h[5] *= expfac;
  }
  if (p_flag[2]) {
    oldlo = domain->boxlo[2]; oldhi = domain->boxhi[2];
    expfac = exp(dto*omega_dot[2]);
    domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
    domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    if (scalexz) h[4] *= expfac;
    if (scaleyz) h[3] *= expfac;
  }

  if (pstyle == TRICLINIC) {
    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac; h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]); h[4] *= expfac;
    }
    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac; h[3] += dto2*(omega_dot[3]*h[2]); h[3] *= expfac;
    }
    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac; h[5] += dto2*(omega_dot[5]*h[1]); h[5] *= expfac;
    }
    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac; h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]); h[4] *= expfac;
    }
  }

  domain->yz = h[3]; domain->xz = h[4]; domain->xy = h[5];

  if (domain->yz < -TILTMAX*domain->yprd ||
      domain->yz >  TILTMAX*domain->yprd ||
      domain->xz < -TILTMAX*domain->xprd ||
      domain->xz >  TILTMAX*domain->xprd ||
      domain->xy < -TILTMAX*domain->xprd ||
      domain->xy >  TILTMAX*domain->xprd)
    error->all(FLERR,"Fix {} has tilted box too far in one step - "
               "periodic cell is too far from equilibrium state", style);

  domain->set_global_box();
  domain->set_local_box();

  if (allremap) domain->lamda2x(nlocal);
  else {
    for (i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit) domain->lamda2x(x[i],x[i]);
  }

  for (auto &ifix : rfix) ifix->deform(1);
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::write_restart(FILE *fp)
{
  nh_langevin_debug("write-restart", style, 0, nullptr);
  int nsize = size_restart_global();
  double *list;
  memory->create(list, nsize, "nh_langevin:list");
  pack_restart_data(list);
  if (comm->me == 0) {
    int size = nsize * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), nsize, fp);
  }
  memory->destroy(list);
}

/* ---------------------------------------------------------------------- */

int FixNHLangevin::size_restart_global()
{
  // Mirrors fix_nh layout but without NHC chains.
  // tstat_flag placeholder + pstat_flag
  // if pstat: omega[6] + omega_dot[6] + vol0 + t0 + mpchain_placeholder(=0)
  //           + deviatoric_flag [+ h0_inv[6]]
  int nsize = 2;
  if (pstat_flag) {
    nsize += 16;   // 6+6+1+1+1(mpchain=0)+1(deviatoric_flag)
    if (deviatoric_flag) nsize += 6;
  }
  return nsize;
}

/* ---------------------------------------------------------------------- */

int FixNHLangevin::pack_restart_data(double *list)
{
  nh_langevin_debug("pack-restart", style, 0, nullptr);
  int n = 0;
  // Match fix_nh restart format so restart files are interoperable.
  list[n++] = tstat_flag;
  // tstat block: if tstat_flag, fix_nh would store mtchain+eta[]+eta_dot[]
  // We store nothing (mtchain=0 equivalent already consumed by tstat_flag=1
  // case in fix_nh::restart — so we must NOT write a chain count here;
  // instead we keep tstat_flag=0 in the restart to avoid confusion).
  // For simplicity we always write 0 here so reading code skips the block.
  // (Override tstat_flag placeholder to 0 to be safe.)
  list[0] = 0;

  list[n++] = pstat_flag;
  if (pstat_flag) {
    for (int i = 0; i < 6; i++) list[n++] = omega[i];
    for (int i = 0; i < 6; i++) list[n++] = omega_dot[i];
    list[n++] = vol0;
    list[n++] = t0;
    list[n++] = 0;   // mpchain placeholder = 0 (no NHC for barostat)
    list[n++] = deviatoric_flag;
    if (deviatoric_flag)
      for (int i = 0; i < 6; i++) list[n++] = h0_inv[i];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::restart(char *buf)
{
  nh_langevin_debug("restart-enter", style, 0, nullptr);
  int n = 0;
  auto *list = (double *) buf;

  int flag = static_cast<int>(list[n++]);
  if (flag) {
    // fix_nh would read: mtchain, eta[], eta_dot[] — skip them
    int m = static_cast<int>(list[n++]);
    n += 2 * m;
  }

  flag = static_cast<int>(list[n++]);
  if (flag) {
    for (int i = 0; i < 6; i++) omega[i]     = list[n++];
    for (int i = 0; i < 6; i++) omega_dot[i] = list[n++];
    vol0 = list[n++];
    t0   = list[n++];
    int m = static_cast<int>(list[n++]);   // mpchain — skip NHC barostat data
    if (pstat_flag && m == 0) {
      // written by us: no etap data
    } else {
      n += 2 * m;   // skip etap[], etap_dot[] from a fix_nh restart
    }
    flag = static_cast<int>(list[n++]);
    if (flag)
      for (int i = 0; i < 6; i++) h0_inv[i] = list[n++];
  }
}

/* ---------------------------------------------------------------------- */

int FixNHLangevin::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (tcomputeflag) { modify->delete_compute(id_temp); tcomputeflag = 0; }
    delete[] id_temp;
    id_temp = utils::strdup(arg[1]);
    temperature = modify->get_compute_by_id(arg[1]);
    if (!temperature)
      error->all(FLERR,"Could not find fix_modify temperature ID {}", arg[1]);
    if (temperature->tempflag == 0)
      error->all(FLERR,"Fix_modify temperature ID {} does not compute temperature", arg[1]);
    if (temperature->igroup != 0 && comm->me == 0)
      error->warning(FLERR,"Temperature for fix modify is not for group all");
    if (pstat_flag) {
      auto *icompute = modify->get_compute_by_id(id_press);
      if (!icompute)
        error->all(FLERR,"Pressure ID {} for fix modify does not exist", id_press);
      icompute->reset_extra_compute_fix(id_temp);
    }
    return 2;
  } else if (strcmp(arg[0],"press") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR,"fix_modify press", error);
    if (!pstat_flag) error->all(FLERR,"Fix_modify press command without a barostat");
    if (pcomputeflag) { modify->delete_compute(id_press); pcomputeflag = 0; }
    delete[] id_press;
    id_press = utils::strdup(arg[1]);
    pressure = modify->get_compute_by_id(arg[1]);
    if (!pressure)
      error->all(FLERR,"Could not find fix_modify pressure ID {}", arg[1]);
    if (pressure->pressflag == 0)
      error->all(FLERR,"Fix_modify pressure ID {} does not compute pressure", arg[1]);
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

double FixNHLangevin::compute_scalar()
{
  // No NHC chain energy.  Return barostat kinetic + PV energy only.
  double energy = 0.0;
  double volume;
  if (dimension == 3) volume = domain->xprd * domain->yprd * domain->zprd;
  else volume = domain->xprd * domain->yprd;

  if (pstat_flag) {
    for (int i = 0; i < 3; i++) {
      if (p_flag[i])
        energy += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i] +
                  p_hydro*(volume-vol0) / (pdim*nktv2p);
    }
    if (pstyle == TRICLINIC)
      for (int i = 3; i < 6; i++)
        if (p_flag[i])
          energy += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i];
    if (deviatoric_flag) energy += compute_strain_energy();
  }
  return energy;
}

/* ---------------------------------------------------------------------- */

double FixNHLangevin::compute_vector(int n)
{
  int ilen;

  if (pstat_flag) {
    if (pstyle == ISO) {
      ilen = 1; if (n < ilen) return omega[n]; n -= ilen;
      ilen = 1; if (n < ilen) return omega_dot[n]; n -= ilen;
    } else if (pstyle == ANISO) {
      ilen = 3; if (n < ilen) return omega[n]; n -= ilen;
      ilen = 3; if (n < ilen) return omega_dot[n]; n -= ilen;
    } else {
      ilen = 6; if (n < ilen) return omega[n]; n -= ilen;
      ilen = 6; if (n < ilen) return omega_dot[n]; n -= ilen;
    }
    if (deviatoric_flag) {
      ilen = 1;
      if (n < ilen) return compute_strain_energy();
      n -= ilen;
    }
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::reset_dt()
{
  dtv    = update->dt;
  dtf    = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4    = 0.25 * update->dt;
  dt8    = 0.125 * update->dt;
  dto    = dthalf;

  if (utils::strmatch(update->integrate_style,"^respa")) {
    auto *respa_ptr = dynamic_cast<Respa *>(update->integrate);
    if (!respa_ptr)
      error->all(FLERR,"Failure to access Respa style {}", update->integrate_style);
    nlevels_respa = respa_ptr->nlevels;
    step_respa    = respa_ptr->step;
    dto = 0.5 * step_respa[0];
  }

  if (pstat_flag)
    pdrag_factor = 1.0 - (update->dt * p_freq_max * drag / 1.0);

  if (tstat_flag)
    tdrag_factor = 1.0 - (update->dt * (1.0/damp_t) * drag / 1.0);
}

/* ---------------------------------------------------------------------- */

void *FixNHLangevin::extract(const char *str, int &dim)
{
  dim = 0;
  if (tstat_flag && strcmp(str,"t_target") == 0) return &t_target;
  if (tstat_flag && strcmp(str,"t_start")  == 0) return &t_start;
  if (tstat_flag && strcmp(str,"t_stop")   == 0) return &t_stop;
  dim = 1;
  if (pstat_flag && strcmp(str,"p_flag")   == 0) return &p_flag;
  if (pstat_flag && strcmp(str,"p_start")  == 0) return &p_start;
  if (pstat_flag && strcmp(str,"p_stop")   == 0) return &p_stop;
  if (pstat_flag && strcmp(str,"p_target") == 0) return &p_target;
  return nullptr;
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::nh_v_press()
{
  double factor[3];
  double **v = atom->v;
  int *mask  = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  factor[0] = exp(-dt4*(omega_dot[0]+mtk_term2));
  factor[1] = exp(-dt4*(omega_dot[1]+mtk_term2));
  factor[2] = exp(-dt4*(omega_dot[2]+mtk_term2));

  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] *= factor[0]; v[i][1] *= factor[1]; v[i][2] *= factor[2];
        if (pstyle == TRICLINIC) {
          v[i][0] += -dthalf*(v[i][1]*omega_dot[5] + v[i][2]*omega_dot[4]);
          v[i][1] += -dthalf* v[i][2]*omega_dot[3];
        }
        v[i][0] *= factor[0]; v[i][1] *= factor[1]; v[i][2] *= factor[2];
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        temperature->remove_bias(i,v[i]);
        v[i][0] *= factor[0]; v[i][1] *= factor[1]; v[i][2] *= factor[2];
        if (pstyle == TRICLINIC) {
          v[i][0] += -dthalf*(v[i][1]*omega_dot[5] + v[i][2]*omega_dot[4]);
          v[i][1] += -dthalf* v[i][2]*omega_dot[3];
        }
        v[i][0] *= factor[0]; v[i][1] *= factor[1]; v[i][2] *= factor[2];
        temperature->restore_bias(i,v[i]);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

// nh_v_temp: kept as a no-op virtual so child classes that override it
// (e.g. fix_nvt/sphere for dipoles) still compile and link correctly.
// In pure Langevin mode the thermostat is applied via langevin_temp().
void FixNHLangevin::nh_v_temp()
{
  // Intentionally empty — Langevin thermostat uses langevin_temp() instead.
  // Subclasses that override this (e.g. fix_nvt/sphere) must also override
  // langevin_temp() or call nh_v_temp() explicitly if they need custom
  // per-particle velocity rescaling beyond the stochastic kick.
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::nve_v()
{
  double dtfm;
  double **v  = atom->v;
  double **f  = atom->f;
  double *rmass = atom->rmass;
  double *mass  = atom->mass;
  int *type   = atom->type;
  int *mask   = atom->mask;
  int nlocal  = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask  = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::compute_sigma()
{
  if (nreset_h0 > 0) {
    bigint delta = update->ntimestep - update->beginstep;
    if (delta % nreset_h0 == 0) {
      if (dimension == 3) vol0 = domain->xprd * domain->yprd * domain->zprd;
      else vol0 = domain->xprd * domain->yprd;
      for (int i = 0; i < 6; i++) h0_inv[i] = domain->h_inv[i];
    }
  }

  sigma[0] =
    vol0*(h0_inv[0]*((p_target[0]-p_hydro)*h0_inv[0] +
                     p_target[5]*h0_inv[5]+p_target[4]*h0_inv[4]) +
          h0_inv[5]*(p_target[5]*h0_inv[0] +
                     (p_target[1]-p_hydro)*h0_inv[5]+p_target[3]*h0_inv[4]) +
          h0_inv[4]*(p_target[4]*h0_inv[0]+p_target[3]*h0_inv[5] +
                     (p_target[2]-p_hydro)*h0_inv[4]));
  sigma[1] =
    vol0*(h0_inv[1]*((p_target[1]-p_hydro)*h0_inv[1]+p_target[3]*h0_inv[3]) +
          h0_inv[3]*(p_target[3]*h0_inv[1]+(p_target[2]-p_hydro)*h0_inv[3]));
  sigma[2] = vol0*(h0_inv[2]*((p_target[2]-p_hydro)*h0_inv[2]));
  sigma[3] =
    vol0*(h0_inv[1]*(p_target[3]*h0_inv[2]) +
          h0_inv[3]*((p_target[2]-p_hydro)*h0_inv[2]));
  sigma[4] =
    vol0*(h0_inv[0]*(p_target[4]*h0_inv[2]) +
          h0_inv[5]*(p_target[3]*h0_inv[2]) +
          h0_inv[4]*((p_target[2]-p_hydro)*h0_inv[2]));
  sigma[5] =
    vol0*(h0_inv[0]*(p_target[5]*h0_inv[1]+p_target[4]*h0_inv[3]) +
          h0_inv[5]*((p_target[1]-p_hydro)*h0_inv[1]+p_target[3]*h0_inv[3]) +
          h0_inv[4]*(p_target[3]*h0_inv[1]+(p_target[2]-p_hydro)*h0_inv[3]));
}

/* ---------------------------------------------------------------------- */

double FixNHLangevin::compute_strain_energy()
{
  double *h = domain->h;
  double d0 =
    sigma[0]*(h[0]*h[0]+h[5]*h[5]+h[4]*h[4]) +
    sigma[5]*(          h[1]*h[5]+h[3]*h[4]) +
    sigma[4]*(                    h[2]*h[4]);
  double d1 =
    sigma[5]*(          h[5]*h[1]+h[4]*h[3]) +
    sigma[1]*(          h[1]*h[1]+h[3]*h[3]) +
    sigma[3]*(                    h[2]*h[3]);
  double d2 =
    sigma[4]*(                    h[4]*h[2]) +
    sigma[3]*(                    h[3]*h[2]) +
    sigma[2]*(                    h[2]*h[2]);
  return 0.5*(d0+d1+d2)/nktv2p;
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::compute_deviatoric()
{
  double *h = domain->h;
  fdev[0] =
    h[0]*(sigma[0]*h[0]+sigma[5]*h[5]+sigma[4]*h[4]) +
    h[5]*(sigma[5]*h[0]+sigma[1]*h[5]+sigma[3]*h[4]) +
    h[4]*(sigma[4]*h[0]+sigma[3]*h[5]+sigma[2]*h[4]);
  fdev[1] =
    h[1]*(              sigma[1]*h[1]+sigma[3]*h[3]) +
    h[3]*(              sigma[3]*h[1]+sigma[2]*h[3]);
  fdev[2] = h[2]*(sigma[2]*h[2]);
  fdev[3] = h[1]*(sigma[3]*h[2]) + h[3]*(sigma[2]*h[2]);
  fdev[4] =
    h[0]*(sigma[4]*h[2]) + h[5]*(sigma[3]*h[2]) + h[4]*(sigma[2]*h[2]);
  fdev[5] =
    h[0]*(sigma[5]*h[1]+sigma[4]*h[3]) +
    h[5]*(sigma[1]*h[1]+sigma[3]*h[3]) +
    h[4]*(sigma[3]*h[1]+sigma[2]*h[3]);
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::compute_temp_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;
  t_target  = t_start + delta * (t_stop - t_start);
  ke_target = tdof * boltz * t_target;
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::compute_press_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;
  p_hydro = 0.0;
  for (int i = 0; i < 3; i++) {
    if (p_flag[i]) {
      p_target[i] = p_start[i] + delta * (p_stop[i] - p_start[i]);
      p_hydro += p_target[i];
    }
  }
  if (pdim > 0) p_hydro /= pdim;
  if (pstyle == TRICLINIC)
    for (int i = 3; i < 6; i++)
      p_target[i] = p_start[i] + delta * (p_stop[i] - p_start[i]);
  if (deviatoric_flag) compute_sigma();
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::nh_omega_dot()
{
  double f_omega, volume;
  if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
  else volume = domain->xprd*domain->yprd;

  if (deviatoric_flag) compute_deviatoric();

  mtk_term1 = 0.0;
  if (mtk_flag) {
    if (pstyle == ISO) {
      mtk_term1 = tdof * boltz * t_current;
      mtk_term1 /= pdim * atom->natoms;
    } else {
      double *mvv_current = temperature->vector;
      for (int i = 0; i < 3; i++)
        if (p_flag[i]) mtk_term1 += mvv_current[i];
      mtk_term1 /= pdim * atom->natoms;
    }
  }

  for (int i = 0; i < 3; i++) {
    if (p_flag[i]) {
      f_omega = (p_current[i]-p_hydro)*volume /
                (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega * dthalf;
      omega_dot[i] *= pdrag_factor;
    }
  }

  mtk_term2 = 0.0;
  if (mtk_flag) {
    for (int i = 0; i < 3; i++)
      if (p_flag[i]) mtk_term2 += omega_dot[i];
    if (pdim > 0) mtk_term2 /= pdim * atom->natoms;
  }

  if (pstyle == TRICLINIC) {
    for (int i = 3; i < 6; i++) {
      if (p_flag[i]) {
        f_omega = p_current[i]*volume/(omega_mass[i] * nktv2p);
        if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
        omega_dot[i] += f_omega * dthalf;
        omega_dot[i] *= pdrag_factor;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNHLangevin::pre_exchange()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double xtiltmax = (0.5+DELTAFLIP)*xprd;
  double ytiltmax = (0.5+DELTAFLIP)*yprd;

  int flipxy, flipxz, flipyz;
  flipxy = flipxz = flipyz = 0;

  if (domain->yperiodic) {
    if (domain->yz < -ytiltmax) {
      domain->yz += yprd; domain->xz += domain->xy; flipyz = 1;
    } else if (domain->yz >= ytiltmax) {
      domain->yz -= yprd; domain->xz -= domain->xy; flipyz = -1;
    }
  }
  if (domain->xperiodic) {
    if (domain->xz < -xtiltmax) { domain->xz += xprd; flipxz = 1; }
    else if (domain->xz >= xtiltmax) { domain->xz -= xprd; flipxz = -1; }
    if (domain->xy < -xtiltmax) { domain->xy += xprd; flipxy = 1; }
    else if (domain->xy >= xtiltmax) { domain->xy -= xprd; flipxy = -1; }
  }

  if (flipxy || flipxz || flipyz) {
    domain->set_global_box();
    domain->set_local_box();
    domain->image_flip(flipxy,flipxz,flipyz);
    double **x  = atom->x;
    imageint *image = atom->image;
    int nlocal  = atom->nlocal;
    for (int i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);
    domain->x2lamda(atom->nlocal);
    irregular->migrate_atoms();
    domain->lamda2x(atom->nlocal);
  }
}

/* ---------------------------------------------------------------------- */

double FixNHLangevin::memory_usage()
{
  double bytes = 0.0;
  if (irregular) bytes += irregular->memory_usage();
  return bytes;
}
