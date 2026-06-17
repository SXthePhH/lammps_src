/* ----------------------------------------------------------------------
  fix_npt_langevin.cpp

   Anisotropic / triclinic Langevin NPT integrator.
   Supports iso, aniso, and tri pressure coupling modes, mirroring
   the pstyle / pcouple logic of fix_nh but replacing the
   Nose-Hoover barostat chain with a simple Langevin damping on omega.

   Voigt index convention (same as fix_nh):
     0=xx  1=yy  2=zz  3=yz  4=xz  5=xy

   Command syntax:
    fix ID group npt_langevin
           temp  T_target  damp_t
           iso   P_target  damp_p  tau_baro  omega_mass_corr
         OR
           aniso P_target  damp_p  tau_baro  omega_mass_corr
         OR
           tri   P_target  damp_p  tau_baro  omega_mass_corr
         [seed S]

   Individual axes can also be specified exactly as in fix_nh:
     x/y/z P_target  damp_p  tau_baro  omega_mass_corr
     yz/xz/xy P_target  damp_p  tau_baro  omega_mass_corr
------------------------------------------------------------------------- */

#include "fix_langevin_npt.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"
#include "memory.h"
#include "output.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

// pstyle
static constexpr int ISO       = 0;
static constexpr int ANISO     = 1;
static constexpr int TRICLINIC = 2;
// pcouple
static constexpr int COUPLE_NONE = 0;
static constexpr int COUPLE_XYZ  = 1;
static constexpr int COUPLE_XY   = 2;
static constexpr int COUPLE_YZ   = 3;
static constexpr int COUPLE_XZ   = 4;

/* ---------------------------------------------------------------------- */

FixNPTLangevin::FixNPTLangevin(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg),
    random(nullptr), id_temp(nullptr), id_press(nullptr)
{
  time_integrate     = 1;
  dynamic_group_allow = 1;
  tcomputeflag       = 1;
  pcomputeflag       = 1;

  if (narg < 8) error->all(FLERR, "Illegal fix npt_langevin command");

  dimension = domain->dimension;

  // ---- defaults ----
  pstyle  = ISO;
  pcouple = COUPLE_NONE;
  for (int i = 0; i < 6; i++) {
    p_flag[i]    = 0;
    p_target[i]  = 0.0;
    p_current[i] = 0.0;
    omega[i]     = 0.0;
    omega_mass[i]= 0.0;
  }
  for (int i = 0; i < 6; i++) {
  omega_exp[i] = (i < 3) ? 1.0 : 0.0;
  }
  damp_p = damp_t = tau_baro = omega_mass_corr = 0.0;
  t_target = t_current = 0.0;

  v_storage = nullptr;
  v_backup = nullptr;
  v_stored = 0;
  nmax = 0;

  // We rely on restoring the physical state (x, v) so ComputePressure will act normally
  // No need to set fix virial flags unless we manually modify virial array.


  seed = 123456789;
  zero_flag = 1;

  // tilt scaling: default mirrors fix_nh (scale tilt with cell if periodic)
  scalexy = scaleyz = scalexz = 0;
  if (domain->yperiodic && domain->xy != 0.0) scalexy = 1;
  if (domain->zperiodic && dimension == 3) {
    if (domain->yz != 0.0) scaleyz = 1;
    if (domain->xz != 0.0) scalexz = 1;
  }

  // ---- parse keywords ----
  int iarg = 3;
  while (iarg < narg) {

    // --- thermostat ---
    if (strcmp(arg[iarg], "temp") == 0) {
      if (iarg+2 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} temp", style), error);
      t_target = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_t   = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      iarg += 3;

    // --- iso: XYZ coupled, single omega (conceptually), but still 3 stored ---
    } else if (strcmp(arg[iarg], "iso") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} iso", style), error);
      double ptgt     = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p          = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro        = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      pcouple = COUPLE_XYZ;
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      p_target[0] = p_target[1] = p_target[2] = ptgt;
      if (dimension == 2) { p_flag[2] = 0; p_target[2] = 0.0; }
      iarg += 5;

    // --- aniso: independent diagonal omegas ---
    } else if (strcmp(arg[iarg], "aniso") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} aniso", style), error);
      double ptgt     = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p          = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro        = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      pcouple = COUPLE_NONE;
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      p_target[0] = p_target[1] = p_target[2] = ptgt;
      if (dimension == 2) { p_flag[2] = 0; p_target[2] = 0.0; }
      iarg += 5;

    // --- tri: all 6 components ---
    } else if (strcmp(arg[iarg], "tri") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} tri", style), error);
      double ptgt     = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p          = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro        = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      pcouple = COUPLE_NONE;
      scalexy = scalexz = scaleyz = 0;  // off-diagonal driven dynamically
      for (int i = 0; i < 6; i++) { p_flag[i] = 1; p_target[i] = (i < 3) ? ptgt : 0.0; }
      if (dimension == 2) {
        p_flag[2] = p_flag[3] = p_flag[4] = 0;
        p_target[2] = p_target[3] = p_target[4] = 0.0;
      }
      iarg += 5;

    // --- individual axes (mirrors fix_nh x/y/z/yz/xz/xy keywords) ---
    } else if (strcmp(arg[iarg], "x") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} x", style), error);
      p_flag[0]    = 1;
      p_target[0]  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p       = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro     = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg], "y") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} y", style), error);
      p_flag[1]    = 1;
      p_target[1]  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p       = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro     = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg], "z") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} z", style), error);
      if (dimension == 2)
        error->all(FLERR, "Invalid fix npt_langevin z for 2d simulation");
      p_flag[2]    = 1;
      p_target[2]  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p       = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro     = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg], "yz") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} yz", style), error);
      if (dimension == 2)
        error->all(FLERR, "Invalid fix npt_langevin yz for 2d simulation");
      p_flag[3]    = 1;
      p_target[3]  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p       = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro     = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      scaleyz      = 0;
      iarg += 5;
    } else if (strcmp(arg[iarg], "xz") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} xz", style), error);
      if (dimension == 2)
        error->all(FLERR, "Invalid fix npt_langevin xz for 2d simulation");
      p_flag[4]    = 1;
      p_target[4]  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p       = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro     = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      scalexz      = 0;
      iarg += 5;
    } else if (strcmp(arg[iarg], "xy") == 0) {
      if (iarg+4 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} xy", style), error);
      p_flag[5]    = 1;
      p_target[5]  = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      damp_p       = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      tau_baro     = utils::numeric(FLERR, arg[iarg+3], false, lmp);
      omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      scalexy      = 0;
      iarg += 5;

    } else if (strcmp(arg[iarg], "seed") == 0) {
      if (iarg+1 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} seed", style), error);
      seed = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg], "zero") == 0) {
      if (iarg+1 >= narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} zero", style), error);
      zero_flag = utils::logical(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;

    } else {
      error->all(FLERR, "Unknown fix npt_langevin keyword: {}", arg[iarg]);
    }
  }

  // ---- determine pstyle ----
  if (p_flag[3] || p_flag[4] || p_flag[5])
    pstyle = TRICLINIC;
  else if (pcouple == COUPLE_XYZ)
    pstyle = ISO;
  else
    pstyle = ANISO;

  // ---- set box_change flags ----
  if (p_flag[0]) box_change |= BOX_CHANGE_X;
  if (p_flag[1]) box_change |= BOX_CHANGE_Y;
  if (p_flag[2]) box_change |= BOX_CHANGE_Z;
  if (p_flag[3]) box_change |= BOX_CHANGE_YZ;
  if (p_flag[4]) box_change |= BOX_CHANGE_XZ;
  if (p_flag[5]) box_change |= BOX_CHANGE_XY;
  restart_pbc = 1;

  gamma_t = 1.0 / damp_t;
  gamma_p = 1.0 / damp_p;

  random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */

FixNPTLangevin::~FixNPTLangevin()
{
  delete random;
  if (tcomputeflag && id_temp) modify->delete_compute(id_temp);
  if (pcomputeflag && id_press) modify->delete_compute(id_press);
  delete[] id_temp;
  delete[] id_press;
  
  if (v_storage) memory->destroy(v_storage);
  if (v_backup)  memory->destroy(v_backup);
}

/* ---------------------------------------------------------------------- */

int FixNPTLangevin::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= END_OF_STEP;    // Needed for thermo hack
  return mask;
}

/* ---------------------------------------------------------------------- */

int FixNPTLangevin::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    
    // 如果之前内部创建了 compute，现在删掉它，因为用户指定了新的
    if (tcomputeflag) {
      modify->delete_compute(id_temp);
      tcomputeflag = 0;
    }
    delete[] id_temp;
    id_temp = utils::strdup(arg[1]);

    // 检查新指定的 compute 是否存在且合法
    auto *temperature = modify->get_compute_by_id(arg[1]);
    if (!temperature)
      error->all(FLERR,"Could not find fix_modify temperature ID {}", arg[1]);

    if (temperature->tempflag == 0)
      error->all(FLERR, "Fix_modify temperature ID {} does not compute temperature", arg[1]);
    
    if (temperature->igroup != 0 && comm->me == 0)
      error->warning(FLERR,"Temperature for fix modify is not for group all");

    // 如果有压控，需要通知 pressure compute 更新它关联的 temp compute
    // Note: this fix assumes pressure control is available; verify pcomputeflag logic if changed.
    if (pcomputeflag) {
      auto *icompute = modify->get_compute_by_id(id_press);
      // id_press 可能也需要检查是否存在
      if (icompute) icompute->reset_extra_compute_fix(id_temp);
    }
    return 2;

  } else if (strcmp(arg[0],"press") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR,"fix_modify press", error);
    
    // 如果之前内部创建了 pressure compute，删掉它
    if (pcomputeflag) {
      modify->delete_compute(id_press);
      pcomputeflag = 0;
    }
    delete[] id_press;
    id_press = utils::strdup(arg[1]);

    auto *pressure = modify->get_compute_by_id(arg[1]);
    if (!pressure) error->all(FLERR,"Could not find fix_modify pressure ID {}", arg[1]);

    if (pressure->pressflag == 0)
      error->all(FLERR,"Fix_modify pressure ID {} does not compute pressure", arg[1]);
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

void FixNPTLangevin::init()
{
  boltz = force->boltz;
  dt    = update->dt;
  dt2   = dt * 0.5;
  dt4   = dt * 0.25;

  v_stored = 0;

  // alpha = 1.0 + 1.0 / (double)atom->natoms;  // MTK correction
  lan_c1_t = exp(-gamma_t * dt);
  lan_c2_t = sqrt((1.0 - lan_c1_t * lan_c1_t) * boltz * t_target);

  // printf("gamma_t: %f lan_c1_t: %f lan_c2_t: %f\n", gamma_t, lan_c1_t, lan_c2_t);
  // printf("mvv2e: %f\n", force->mvv2e);

  lan_c1_p = exp(-gamma_p * dt);
  lan_c2_p = sqrt((1.0 - lan_c1_p * lan_c1_p) * boltz * t_target);
  lan_c1_p_2 = exp(-gamma_p * dt2);
  lan_c2_p_2 = sqrt((1.0 - lan_c1_p_2 * lan_c1_p_2) * boltz * t_target);

  // barostat mass: W = corr * (N+1) * kT * tau^2  for all components
  // off-diagonal components (3-5) use the same mass formula
  double base_mass = omega_mass_corr * ((double)atom->natoms * 3 + 1.0)
                     * boltz * t_target * tau_baro * tau_baro;   // here default no loss of degrees of freedom
  for (int i = 0; i < 6; i++)
    omega_mass[i] = base_mass;

  temperature = modify->get_compute_by_id(id_temp);
  pressure    = modify->get_compute_by_id(id_press);

  if (force->kspace) kspace_flag = 1;
  else kspace_flag = 0;
}

/* ---------------------------------------------------------------------- */

void FixNPTLangevin::setup(int /*vflag*/)
{
  if (temperature) {
    t_current = temperature->compute_scalar();
    tdof = temperature->dof;
    printf("tdof: %f\n", tdof);
  }

  if (pressure) {
    if (pstyle == ISO) pressure->compute_scalar();
    else               pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep + 1);
  }
  mat_exp(dt2);  // now using the positive sign for position
}

/* ---------------------------------------------------------------------- */

void FixNPTLangevin::reset_dt()
{
  dt  = update->dt;
  dt2 = dt * 0.5;
  dt4 = dt * 0.25;
}

/* ----------------------------------------------------------------------
   Taylor series for sinh(x)/x — accurate for small |x|
---------------------------------------------------------------------- */

double FixNPTLangevin::fast_sinhc(double x)
{
  return 1.0 + (x*x)/6.0
             + (x*x*x*x)/120.0
             + (x*x*x*x*x*x)/5040.0
             + (x*x*x*x*x*x*x*x)/362880.0
             + (x*x*x*x*x*x*x*x*x*x)/39916800.0;
}

/* ----------------------------------------------------------------------
   Populate p_current[6] from pressure compute.

   pressure->vector index order (LAMMPS compute pressure):
     [0]=xx [1]=yy [2]=zz [3]=xy [4]=xz [5]=yz

   fix_nh Voigt order:  xx=0 yy=1 zz=2 yz=3 xz=4 xy=5
   So we must swap: p_current[3] <- tensor[5]  (yz)
                    p_current[4] <- tensor[4]  (xz)
                    p_current[5] <- tensor[3]  (xy)

   For ISO pstyle use scalar; for ANISO/TRICLINIC use vector.
   Coupling modes average the appropriate diagonal components.
---------------------------------------------------------------------- */

void FixNPTLangevin::couple()
{
  if (pstyle == ISO) {
    p_current[0] = p_current[1] = p_current[2] = pressure->scalar;
    return;
  }

  double *tensor = pressure->vector;

  // diagonal components with coupling
  if (pcouple == COUPLE_XYZ) {
    double ave = (tensor[0] + tensor[1] + tensor[2]) / 3.0;
    p_current[0] = p_current[1] = p_current[2] = ave;
  } else if (pcouple == COUPLE_XY) {
    double ave = 0.5 * (tensor[0] + tensor[1]);
    p_current[0] = p_current[1] = ave;
    p_current[2] = tensor[2];
  } else if (pcouple == COUPLE_YZ) {
    double ave = 0.5 * (tensor[1] + tensor[2]);
    p_current[1] = p_current[2] = ave;
    p_current[0] = tensor[0];
  } else if (pcouple == COUPLE_XZ) {
    double ave = 0.5 * (tensor[0] + tensor[2]);
    p_current[0] = p_current[2] = ave;
    p_current[1] = tensor[1];
  } else {
    p_current[0] = tensor[0];
    p_current[1] = tensor[1];
    p_current[2] = tensor[2];
  }

  // off-diagonal components (TRICLINIC only)
  // swap from compute-pressure order [xy=3,xz=4,yz=5]
  // to Voigt order [yz=3,xz=4,xy=5]
  if (pstyle == TRICLINIC) {
    p_current[3] = tensor[5];  // yz
    p_current[4] = tensor[4];  // xz
    p_current[5] = tensor[3];  // xy
  }
}

/* ----------------------------------------------------------------------
   Half-step update of all active barostat velocities omega[6].

   For diagonal components (0-2), the MTK-corrected EOM is:
     d(omega_i)/dt = V/W_i * (P_ii - P_hydro) / nktv2p
                   + kT / W_i                        (MTK term)

   Note: we use p_hydro (average target) for the diagonal driving force,
   matching fix_nh's nh_omega_dot which subtracts p_hydro not p_target[i].

   For off-diagonal components (3-5), the EOM is simpler — no MTK term,
   no hydrostatic offset, just:
     d(omega_i)/dt = V/W_i * P_ii / nktv2p
---------------------------------------------------------------------- */

void FixNPTLangevin::update_omega()
{
  double volume = domain->xprd * domain->yprd * domain->zprd;
  double nktv2p = force->nktv2p;
  // double kT     = tdof * boltz * t_current/(atom->natoms * 3);  // per-particle kinetic energy
  double kT     = boltz * t_current;  // per-particle kinetic energy, not scaled by dof or natoms, since MTK correction already accounts for that

  // compute hydrostatic target (average over active diagonal components)
  p_hydro = 0.0;
  int pdim = 0;

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) { p_hydro += p_target[i]; pdim++; }
  if (pdim > 0) p_hydro /= pdim;

  // diagonal: MTK-corrected
  for (int i = 0; i < 3; i++) {
    if (!p_flag[i]) continue;
    double f = volume / omega_mass[i] * (p_current[i] - p_hydro) / nktv2p
             + kT / omega_mass[i];  // MTK correction
    if (pstyle ==ISO) f *= 3.0;  // extra factor of 3 for iso since each omega drives all 3 axes
    omega[i] += dt2 * f;
  }

  // off-diagonal: no MTK, target is zero (deviatoric stress)
  if (pstyle == TRICLINIC) {
    for (int i = 3; i < 6; i++) {
      if (!p_flag[i]) continue;
      double f = volume / omega_mass[i] * p_current[i] / nktv2p;
      omega[i] += dt2 * f;
      // printf("omega[%d]: %f\n", i, omega[i]);
    }
  }
}

/* ----------------------------------------------------------------------
   Half-step velocity update with anisotropic barostat coupling.

   For each Cartesian direction d, the scaling uses omega[d]:
     factor_damp[d] = exp(-omega[d] * alpha * dt2)
     factor_kick[d] = exp(+omega[d] * alpha * dt4)
     sinhc[d]       = sinhc(alpha * omega[d] * dt4)

   For TRICLINIC, there are additional off-diagonal velocity terms
   (mirrors fix_nh nh_v_press):
     v_x -= dt2 * (v_y * omega[5] + v_z * omega[4])
     v_y -= dt2 *  v_z * omega[3]
   These are applied sandwiched between the diagonal scalings.
---------------------------------------------------------------------- */

void FixNPTLangevin::scale_v()
{
  double **v   = atom->v;
  int nlocal   = atom->nlocal;
  double alpha_factor[3];
  mat_exp(-dt2); // using negative sign for velocity update

  // Use negative omega exponentials (mode 1)
  double *w_exp = omega_exp;
  for (int i = 0; i < 6; i++) {
    // printf("omega_exp[%d]: %f\n", i, w_exp[i]);
  }
  // get the trace of omega
  double omega_trace = omega[0] + omega[1] + omega[2];
  double factor = exp(-omega_trace/ (3*atom->natoms) * dt2);
  for (int i = 0; i < nlocal; i++){
    double v0 = v[i][0];
    double v1 = v[i][1];
    double v2 = v[i][2];

    // Apply matrix exponential scaling (exp(-Omega * dt/2))
    v[i][0] = w_exp[0]*v0 + w_exp[5]*v1 + w_exp[4]*v2;
    v[i][1] = w_exp[1]*v1 + w_exp[3]*v2;
    v[i][2] = w_exp[2]*v2;
    
    // Apply MTK correction factor
    v[i][0] *= factor;
    v[i][1] *= factor;
    v[i][2] *= factor;

  }
}

void FixNPTLangevin::update_v()
{
  // Scheme A: velocity half-step.
  // The barostat scaling on velocities is exp(-omega[d]*alpha*dt2) per axis,
  // plus the off-diagonal coupling terms for triclinic.
  // The force kick is a plain dtf = 0.5*dt*ftm2v, no sinhc needed.
  double **v   = atom->v;
  double **f   = atom->f;
  double *mass = atom->mass;
  double *rmass= atom->rmass;
  int nlocal   = atom->nlocal;
  double ftm2v = force->ftm2v;
  double dtf   = 0.5 * dt * ftm2v;

  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      double im = 1.0 / rmass[i];
      v[i][0] += dtf * f[i][0] * im;
      v[i][1] += dtf * f[i][1] * im;
      v[i][2] += dtf * f[i][2] * im;
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      double im = 1.0 / mass[atom->type[i]];
      v[i][0] += dtf * f[i][0] * im;
      v[i][1] += dtf * f[i][1] * im;
      v[i][2] += dtf * f[i][2] * im;
    }
  }
}

/* ----------------------------------------------------------------------
   Half-step position update with anisotropic barostat.

   Per-axis:
     factor2[d] = exp(omega[d] * dt2)
     factor4[d] = exp(omega[d] * dt4)
     sinhc[d]   = sinhc(omega[d] * dt4)
     x_d = factor2[d] * x_d + dt2 * factor4[d] * sinhc[d] * v_d
---------------------------------------------------------------------- */

void FixNPTLangevin::update_x()
{
  // Scheme A: remap() handles atom scaling with the box via x2lamda/lamda2x.
  // Here we only need a plain NVE full-step position update.
  double **x = atom->x;
  double **v = atom->v;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    x[i][0] += dt2 * v[i][0];
    x[i][1] += dt2 * v[i][1];
    x[i][2] += dt2 * v[i][2];
  }
}

/* ----------------------------------------------------------------------
   Half-step box remap using current omega[6].

   Strategy mirrors fix_nh::remap():
     1. Convert atoms to fractional (lambda) coords
     2. Update h-matrix elements (diagonal + off-diagonal for triclinic)
     3. Rebuild global/local box
     4. Convert atoms back to Cartesian

   For triclinic, the off-diagonal h elements (h[3]=yz, h[4]=xz, h[5]=xy)
   are updated with the same symmetrised integrator as fix_nh — a
   quadruple exp-sandwich that preserves time-reversibility:

     h[i] = exp(A*dt/8) * [exp(A*dt/8)*h[i] + dt/4*coupling] * exp(A*dt/8)
     (first half), then diagonal scale, then second half.

   For ANISO, only diagonal elements are updated, but atoms must still
   be converted to lambda coords because each axis scales differently.
---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Helper functions for exp_omega to ensure numerical stability
   when eigenvalues are degenerate or nearly so.
------------------------------------------------------------------------- */

static inline double phi(double x)
{
  const double eps = 1e-8;

  if (fabs(x) < eps) {
    // 4th order Taylor
    return 1.0 + x*0.5 + x*x/6.0 + x*x*x/24.0 + x*x*x*x/120.0;
  }

  return expm1(x) / x;
}


static inline double diff_val(double x, double y)
{
  double d = x - y;

  if (fabs(d) < 1e-8) {
    // limit → exp(x)
    return exp(x);
  }

  // exp(y) * phi(x-y)
  return exp(y) * phi(d);
}

static double diff2_val(double x, double y, double z) {
  double d = y - z;
  double val1 = diff_val(x, y);
  double val2 = diff_val(x, z);

  if (fabs(d) < 1.0e-6) {
    // Derivative of diff_val(x, y) with respect to y
    // d/dy [ (e^x - e^y)/(x - y) ]
    // = e^y * (e^(x-y) - 1 - (x-y)) / (x-y)^2
    double v = x - y;
    if (fabs(v) < 1.0e-6) // Taylor for (e^v - 1 - v)/v^2 -> 1/2 + v/6 + v^2/24
      return exp(y) * (0.5 + v/6.0 + v*v/24.0);
    return exp(y) * (exp(v) - 1.0 - v) / (v * v);
  }
  return (val1 - val2) / d;
}

// stable divided difference

void FixNPTLangevin::mat_exp(double delta_t)
{
  double omega_dt[6];
  for (int i = 0; i < 6; i++)
    omega_dt[i] = omega[i] * delta_t;

  // diagonal exponentials
  omega_exp[0] = exp(omega_dt[0]);
  omega_exp[1] = exp(omega_dt[1]);
  omega_exp[2] = exp(omega_dt[2]);

  if (pstyle != TRICLINIC)
    return;

  // first-order divided differences
  double d01 = diff_val(omega_dt[0], omega_dt[1]);
  double d12 = diff_val(omega_dt[1], omega_dt[2]);
  double d02 = diff_val(omega_dt[0], omega_dt[2]);

  omega_exp[5] = omega_dt[5] * d01;
  omega_exp[3] = omega_dt[3] * d12;

  double term1 = omega_dt[4] * d02;

  // ---- stable second divided difference part ----
  double denom = omega_dt[1] - omega_dt[2];
  double term2;

  if (fabs(denom) < 1e-8) {
    // ω1 ≈ ω2 退化情况
    // 使用极限：
    // (diff01 - diff02)/(ω1-ω2) → ∂/∂ω1 diff(ω0, ω1)

    double x = omega_dt[0];
    double y = omega_dt[1];

    // derivative of diff(x,y) wrt y
    // diff = exp(y) * phi(x-y)
    // d/dy = exp(y)*phi + exp(y)*phi'*(-1)

    double d = x - y;
    double expy = exp(y);

    double ph = phi(d);

    double ph_prime;
    if (fabs(d) < 1e-8) {
      // derivative of phi at 0
      ph_prime = 0.5 + d/3.0;
    } else {
      ph_prime =
        ( (d*exp(d) - expm1(d)) / (d*d) );
    }

    double derivative = expy * ph - expy * ph_prime;

    term2 = omega_dt[3] * omega_dt[5] * derivative;
  }
  else {
    term2 = omega_dt[3] * omega_dt[5]
          * (d01 - d02) / denom;
  }

  omega_exp[4] = term1 + term2;
}


void FixNPTLangevin::remap()
{
  int nlocal = atom->nlocal;
  double *h  = domain->h;
  mat_exp(dt2);  // now using the positive sign for position

  // 1. Convert atoms to fractional coords using the OLD box
  domain->x2lamda(nlocal);

  // 2. Update box dimensions using the matrix exponential M = exp(Omega * dt/2)
  // We use mode 0 (positive argument) for box expansion.
  double *w_exp = omega_exp;

  // H_new = M * H_old
  double h0 = h[0];
  double h1 = h[1];
  double h2 = h[2];
  double h3 = h[3]; // yz
  double h4 = h[4]; // xz
  double h5 = h[5]; // xy
  
  // Diagonal update
  h[0] = w_exp[0] * h0;
  h[1] = w_exp[1] * h1;
  h[2] = w_exp[2] * h2;
  
  // Off-diagonal update
  if (pstyle == TRICLINIC) {
    // h5 (xy) = m0*h5 + m5*h1
    h[5] = w_exp[0]*h5 + w_exp[5]*h1;
    
    // h4 (xz) = m0*h4 + m5*h3 + m4*h2 -- Wait, matrix multiply:
    // Row 0 of M is [m0 m5 m4]
    // Col 2 of H is [h4 h3 h2]^T
    // Product (0,2) = m0*h4 + m5*h3 + m4*h2. Correct.
    h[4] = w_exp[0]*h4 + w_exp[5]*h3 + w_exp[4]*h2;
    
    // h3 (yz) = m1*h3 + m3*h2
    h[3] = w_exp[1]*h3 + w_exp[3]*h2;
    
    // Update domain tilt factors to match h
    domain->yz = h[3];
    domain->xz = h[4];
    domain->xy = h[5];
  }
  
  // Update box boundaries (centered scaling)
  // X
  double oldlo = domain->boxlo[0]; 
  double oldhi = domain->boxhi[0];
  double center = 0.5*(oldlo+oldhi);
  double half = 0.5*(h[0]);
  domain->boxlo[0] = center - half;
  domain->boxhi[0] = center + half;

  // Y
  oldlo = domain->boxlo[1]; 
  oldhi = domain->boxhi[1];
  center = 0.5*(oldlo+oldhi);
  half = 0.5*(h[1]);
  domain->boxlo[1] = center - half;
  domain->boxhi[1] = center + half;
  
  // Z
  oldlo = domain->boxlo[2]; 
  oldhi = domain->boxhi[2];
  center = 0.5*(oldlo+oldhi);
  half = 0.5*(h[2]);
  domain->boxlo[2] = center - half;
  domain->boxhi[2] = center + half;

  // 3. Rebuild global/local box
  domain->set_global_box();
  domain->set_local_box();

  // 4. Convert atoms back to Cartesian using the NEW box
  // This effectively applies the deformation x_new = H_new * H_old^-1 * x_old
  domain->lamda2x(nlocal);
  
  // Also deform rigid bodies if any (from fix_nh logic)
  // for (auto &ifix : rfix) ifix->deform(0); 
  // (Not in original code provided, but good to keep in mind if adapting)
}

/* ----------------------------------------------------------------------
   Langevin thermostat O-step on particle velocities.
---------------------------------------------------------------------- */

void FixNPTLangevin::langevin_temp()
{
  double **v   = atom->v;
  double *mass = atom->mass;
  double *rmass= atom->rmass;
  double mvv2e = force->mvv2e;
  int nlocal   = atom->nlocal;
  double fsum[4] = {0.0, 0.0, 0.0, 0.0};
  double fsumall[4] = {0.0, 0.0, 0.0, 0.0};

  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      double inv_sqrt_m = 1.0 / sqrt(rmass[i] * mvv2e);
      double fran0 = lan_c2_t * random->gaussian() * inv_sqrt_m;
      double fran1 = lan_c2_t * random->gaussian() * inv_sqrt_m;
      double fran2 = lan_c2_t * random->gaussian() * inv_sqrt_m;
      double mass_i = rmass[i];
      if (zero_flag) {
        fsum[0] += mass_i * fran0;
        fsum[1] += mass_i * fran1;
        fsum[2] += mass_i * fran2;
        fsum[3] += mass_i;
      }
      v[i][0] = lan_c1_t * v[i][0] + fran0;
      v[i][1] = lan_c1_t * v[i][1] + fran1;
      v[i][2] = lan_c1_t * v[i][2] + fran2;
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      double mass_i = mass[atom->type[i]];
      double inv_sqrt_m = 1.0 / sqrt(mass_i * mvv2e);
      double fran0 = lan_c2_t * random->gaussian() * inv_sqrt_m;
      double fran1 = lan_c2_t * random->gaussian() * inv_sqrt_m;
      double fran2 = lan_c2_t * random->gaussian() * inv_sqrt_m;
      if (zero_flag) {
        fsum[0] += mass_i * fran0;
        fsum[1] += mass_i * fran1;
        fsum[2] += mass_i * fran2;
        fsum[3] += mass_i;
      }
      v[i][0] = lan_c1_t * v[i][0] + fran0;
      v[i][1] = lan_c1_t * v[i][1] + fran1;
      v[i][2] = lan_c1_t * v[i][2] + fran2;
    }
  }

  if (zero_flag) {
    MPI_Allreduce(fsum, fsumall, 4, MPI_DOUBLE, MPI_SUM, world);
    double inv_tmass = 1.0 / fsumall[3];
    double correction[3] = {fsumall[0] * inv_tmass, fsumall[1] * inv_tmass, fsumall[2] * inv_tmass};

    for (int i = 0; i < nlocal; i++) {
      v[i][0] -= correction[0];
      v[i][1] -= correction[1];
      v[i][2] -= correction[2];
    }
  }
}

/* ----------------------------------------------------------------------
   Langevin O-step on barostat velocities omega[6].
   All 6 components generated on rank 0 and broadcast as one array.
---------------------------------------------------------------------- */

void FixNPTLangevin::langevin_press()
{
  double kicks[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  if (comm->me == 0) {
    if (pcouple == COUPLE_XYZ) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[1] = kicks[2] = kick;
    } else if (pcouple == COUPLE_XY) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[1] = kick;
      if (p_flag[2]) kicks[2] = lan_c2_p * random->gaussian() / sqrt(omega_mass[2]);
    } else if (pcouple == COUPLE_YZ) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[1]);
      kicks[1] = kicks[2] = kick;
      if (p_flag[0]) kicks[0] = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
    } else if (pcouple == COUPLE_XZ) {
      double kick = lan_c2_p * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[2] = kick;
      if (p_flag[1]) kicks[1] = lan_c2_p * random->gaussian() / sqrt(omega_mass[1]);
    } else {
      for (int i = 0; i < 6; i++) {
        if (p_flag[i])
          kicks[i] = lan_c2_p * random->gaussian() / sqrt(omega_mass[i]);
      }
    }
  }

  MPI_Bcast(kicks, 6, MPI_DOUBLE, 0, world);

  for (int i = 0; i < 6; i++) {
    if (p_flag[i])
      omega[i] = lan_c1_p * omega[i] + kicks[i];
  }
}

 /* ----------------------------------------------------------------------
   Compute temperature manually from particle velocities, for testing/debugging.
   This should match the value from the temperature compute (with MTK correction).
---------------------------------------------------------------------- */

double FixNPTLangevin::compute_temp_manual()
{
  double **v   = atom->v;
  double *mass = atom->mass;
  double *rmass= atom->rmass;
  int nlocal   = atom->nlocal;
  double mvv2e = force->mvv2e;   // LJ 单位下 = 1
  double boltz = force->boltz;   // LJ 单位下 = 1

  double ke = 0.0;
  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      ke += rmass[i] * (v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
  } else {
    for (int i = 0; i < nlocal; i++) {
      double m = mass[atom->type[i]];
      ke += m * (v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
    }
  }

  ke *= mvv2e;                        // LJ 下无效果，但保留单位一致性
  int dof_temp = 3 * nlocal - 3;           // 减去整体平动 3 个自由度
  return ke / (dof_temp * boltz);
}

/* ---------------------------------------------------------------------- */

void FixNPTLangevin::initial_integrate(int /*vflag*/)
{
  // B: half-step velocity update
  scale_v();
  update_v();
  // If we hacked the velocities at end_of_step (for thermo output to match t_initial),
  // we MUST restore them now before any physics happens!
  // if (v_stored) {
  //   if (!v_storage) error->one(FLERR, "FixNPTLangevin: v_storage missing!");
    
  //   double **v = atom->v;
  //   int nlocal = atom->nlocal;
  //   for (int i = 0; i < nlocal; i++) {
  //       v[i][0] = v_storage[i][0];
  //       v[i][1] = v_storage[i][1];
  //       v[i][2] = v_storage[i][2];
  //   }
  //   v_stored = 0;
    
  //   // Also clear the fake virial contribution we added for thermo
  //   for (int k = 0; k < 6; k++) virial[k] = 0.0;
  // }

  // B: half-step velocity update
  update_v();
  scale_v();
  // measure T and P after velocity half-step
  if (temperature) {
    t_current = temperature->compute_scalar();
    tdof = temperature->dof;
  }
  
  if (pressure) {
    if (pstyle == ISO) pressure->compute_scalar();
    else               pressure->compute_vector();
    couple(); // this assign p_current from pressure compute
    pressure->addstep(update->ntimestep + 1);
  } else {
    // If no pressure compute defined, couple does nothing or crashes.
    // Assuming fix initialization ensures pressure is set if needed.
  }
  
  // half-step barostat omega update
  update_omega();
  // A: half-step box remap
  remap();
  // A: full-step position update
  update_x();

  // O: Langevin kicks on particles and barostat

  langevin_temp();
  langevin_press();
  // A: full-step position update (second half)
  update_x();
  // A: half-step box remap (second half)
  remap();
  if (kspace_flag) force->kspace->setup();
}

/* ---------------------------------------------------------------------- */

void FixNPTLangevin::final_integrate()
{
  // measure fresh T and P from newly computed forces (Physics requires this!)
  if (temperature) {
    t_current = temperature->compute_scalar();
    tdof = temperature->dof;
  }

  if (pressure) {
    if (pstyle == ISO) pressure->compute_scalar();
    else               pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep + 1);
  }

  // // Save the full velocity state v(t+dt/2) for later restoration at end_of_step
  // // Only execute this when thermo output is active for this step
  // if (output->next_thermo == update->ntimestep) {
  //   if (atom->nmax > nmax) {
  //     nmax = atom->nmax;
  //     // We reallocate both here since nmax tracks both
  //     memory->grow(v_storage, nmax, 3, "fix_npt:v_storage");
  //     memory->grow(v_backup, nmax, 3, "fix_npt:v_backup");
  //   }

    // double **v = atom->v;

    // int nlocal = atom->nlocal;
    // for (int i = 0; i < nlocal; i++) {
    //     v_backup[i][0] = v[i][0];
    //     v_backup[i][1] = v[i][1];
    //     v_backup[i][2] = v[i][2];
    // }
  // }
  
  // half-step barostat omega update using fresh P
  update_omega();

}

/* ---------------------------------------------------------------------- */

void FixNPTLangevin::end_of_step()
{
//   // "Output Hack" Revised:
//   // To satisfy the requirement of recording correct "internal distribution" and "virial data"
//   // corresponding to the beginning of final_integrate, we physically restore the velocity state v(t+dt/2).
//   // 
//   // 1. Save current v(t+dt) to v_storage (to be restored in next initial_integrate).
//   // 2. Overwrite v with v_backup (v(t+dt/2)).
//   // 3. Do NOT modify virial manually; the position/force state is identical to start of final_integrate,
//   //    so ComputePressure will yield the correct virial contribution automatically.
  
//   if (output->next_thermo == update->ntimestep) {
//     if (atom->nmax > nmax) {
//       nmax = atom->nmax;
//       memory->grow(v_storage, nmax, 3, "fix_npt:v_storage");
//       memory->grow(v_backup, nmax, 3, "fix_npt:v_backup");
//     }
  
//     double **v = atom->v;
//     int nlocal = atom->nlocal;
  
//     // 1. Save valid v(t+dt)
//     for (int i=0; i<nlocal; i++) {
//         v_storage[i][0] = v[i][0];
//         v_storage[i][1] = v[i][1];
//         v_storage[i][2] = v[i][2];
//     }
//     v_stored = 1;

//     // 2. Restore v(t+dt/2)
//     if (v_backup) {
//         for (int i=0; i<nlocal; i++) {
//            v[i][0] = v_backup[i][0];
//            v[i][1] = v_backup[i][1];
//            v[i][2] = v_backup[i][2];
//         }
//     }
//   }

//   // Force re-computation of Temperature/Pressure?
//   // Thermo will ask for these.
//   // If thermo uses a distinct compute, it will calculate from current v (which is now v_backup) -> Correct.
//   // If thermo uses fix's compute, it might use cached value from start of final_integrate -> Correct.
//   // We can force clear the cache to be safe, but LAMMPS API doesn't easily support "un-invoking".
//   // However, since we matched the state, the cached value is consistent with current state anyway.
}
