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

#include "fix_nh_middle.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

enum{SIDE,  MIDDLE};
enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};

FixNHMiddle::ArgList FixNHMiddle::filter_middle_args(int narg, char **arg)
{
  FixNHMiddle::ArgList filtered;
  filtered.storage.reserve(narg);

  auto append = [&filtered](const char *value) { filtered.storage.emplace_back(value); };

  for (int i = 0; i < MIN(narg,3); i++) append(arg[i]);

  // FixNH is the base Nose-Hoover parser.  It does not know the extra
  // keywords added by this middle-scheme fix, so remove those keywords before
  // calling the base constructor.
  int iarg = 3;
  while (iarg < narg) {
    if ((strcmp(arg[iarg],"thermostat") == 0 || strcmp(arg[iarg],"barostat") == 0) &&
        iarg+2 < narg && utils::is_double(arg[iarg+2])) {
      iarg += 3;
    } else if (strcmp(arg[iarg],"thermostat") == 0 || strcmp(arg[iarg],"barostat") == 0 ||
               strcmp(arg[iarg],"seed") == 0 || strcmp(arg[iarg],"zero") == 0) {
      iarg += 2;
    } else if (strcmp(arg[iarg],"integrator") == 0) {
      iarg += 2;
    } else {
      append(arg[iarg]);
      iarg++;
    }
  }

  filtered.argv.reserve(filtered.storage.size());
  for (auto &entry : filtered.storage) filtered.argv.push_back(entry.data());
  return filtered;
}

/* ---------------------------------------------------------------------- */

FixNHMiddle::FixNHMiddle(LAMMPS *lmp, int narg, char **arg) :
  FixNHMiddle(lmp, narg, arg, filter_middle_args(narg,arg)) {}

FixNHMiddle::FixNHMiddle(LAMMPS *lmp, int narg, char **arg, ArgList &&filtered) :
  FixNH(lmp, static_cast<int>(filtered.argv.size()), filtered.argv.data()),
  gamma_t(0.0), gamma_p(0.0), damp_t(0.0), damp_p(0.0),
  lan_c1_t(1.0), lan_c2_t(0.0), lan_c1_t_2(1.0), lan_c2_t_2(0.0),
  lan_c1_p(1.0), lan_c2_p(0.0), lan_c1_p_2(1.0), lan_c2_p_2(0.0),
  random(nullptr), seed(12345678), zero_flag(1), integrator(MIDDLE), nh_temp_flag(1),
  nh_press_flag(1), langevin_temp_damp_flag(0), langevin_press_damp_flag(0)
{
  parse_middle_args(narg,arg);
  if (damp_t > 0.0) gamma_t = 1.0 / damp_t;
  if (damp_p > 0.0) gamma_p = 1.0 / damp_p;
  random = new RanMars(lmp,seed);
}

/* ---------------------------------------------------------------------- */

FixNHMiddle::~FixNHMiddle()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::parse_middle_args(int narg, char **arg)
{
  // Parse only the keywords that are new or reinterpreted by FixNHMiddle.
  // Standard FixNH keywords are still parsed by the base class constructor.
  nh_temp_flag = 1;
  nh_press_flag = 1;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"thermostat") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} thermostat", style), error);
      if (strcmp(arg[iarg+1],"nh") == 0) nh_temp_flag = 1;
      else if (strcmp(arg[iarg+1],"langevin") == 0) {
        if (iarg+3 > narg)
          utils::missing_cmd_args(FLERR, fmt::format("fix {} thermostat langevin", style), error);
        if (!utils::is_double(arg[iarg+2]))
          error->all(FLERR, "Fix {} thermostat langevin requires a numeric relaxation time", style);
        nh_temp_flag = 0;
        langevin_temp_damp_flag = 1;
        damp_t = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        iarg++;
      }
      else error->all(FLERR, "Illegal fix {} thermostat option: {}", style, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg],"barostat") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} barostat", style), error);
      if (strcmp(arg[iarg+1],"nh") == 0) nh_press_flag = 1;
      else if (strcmp(arg[iarg+1],"langevin") == 0) {
        if (iarg+3 > narg)
          utils::missing_cmd_args(FLERR, fmt::format("fix {} barostat langevin", style), error);
        if (!utils::is_double(arg[iarg+2]))
          error->all(FLERR, "Fix {} barostat langevin requires a numeric relaxation time", style);
        nh_press_flag = 0;
        langevin_press_damp_flag = 1;
        damp_p = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        iarg++;
      }
      else error->all(FLERR, "Illegal fix {} barostat option: {}", style, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+4 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} temp", style), error);
      iarg += 4;

    } else if (strcmp(arg[iarg],"integrator") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} integrator", style), error);
      if (strcmp(arg[iarg+1],"side") == 0) integrator = SIDE;
      else if (strcmp(arg[iarg+1],"middle") == 0) integrator = MIDDLE;
      else error->all(FLERR, "Illegal fix {} integrator option: {}", style, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} seed", style), error);
      seed = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"zero") == 0) {
      if (iarg+2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} zero", style), error);
      zero_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"x") == 0 || strcmp(arg[iarg],"y") == 0 ||
               strcmp(arg[iarg],"z") == 0 || strcmp(arg[iarg],"xy") == 0 ||
               strcmp(arg[iarg],"xz") == 0 || strcmp(arg[iarg],"yz") == 0 ||
               strcmp(arg[iarg],"iso") == 0 || strcmp(arg[iarg],"aniso") == 0 ||
               strcmp(arg[iarg],"tri") == 0 ||
               strcmp(arg[iarg],"temp") == 0) {
      iarg += 4;
    } else if (strcmp(arg[iarg],"fixedpoint") == 0) {
      iarg += 4;
    } else if (strcmp(arg[iarg],"disc") == 0) {
      iarg++;
    } else if (strcmp(arg[iarg],"erate") == 0 || strcmp(arg[iarg],"strain") == 0) {
      iarg += 3;
    } else if (strcmp(arg[iarg],"ext") == 0 || strcmp(arg[iarg],"couple") == 0 ||
               strcmp(arg[iarg],"drag") == 0 || strcmp(arg[iarg],"ptemp") == 0 ||
               strcmp(arg[iarg],"dilate") == 0 || strcmp(arg[iarg],"tchain") == 0 ||
               strcmp(arg[iarg],"pchain") == 0 || strcmp(arg[iarg],"mtk") == 0 ||
               strcmp(arg[iarg],"tloop") == 0 || strcmp(arg[iarg],"ploop") == 0 ||
               strcmp(arg[iarg],"nreset") == 0 || strcmp(arg[iarg],"scalexy") == 0 ||
               strcmp(arg[iarg],"scalexz") == 0 || strcmp(arg[iarg],"scaleyz") == 0 ||
               strcmp(arg[iarg],"flip") == 0 || strcmp(arg[iarg],"update") == 0 ||
               strcmp(arg[iarg],"psllod") == 0 || strcmp(arg[iarg],"peculiar") == 0 ||
               strcmp(arg[iarg],"kick") == 0) {
      iarg += 2;
    } else {
      iarg++;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::init()
{
  FixNH::init();
  apply_zero_dof_mode();
  update_langevin_coefficients();
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::setup(int vflag)
{
  FixNH::setup(vflag);
  apply_zero_dof_mode();
  update_langevin_coefficients();
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::reset_dt()
{
  FixNH::reset_dt();
  update_langevin_coefficients();
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::apply_zero_dof_mode()
{
  if (!temperature || nh_temp_flag) return;

  if (zero_flag) {
    temperature->reset_extra_dof();
    temperature->setup();
    return;
  }

  char *args[2];
  args[0] = const_cast<char *>("extra/dof");
  args[1] = const_cast<char *>("0");
  temperature->modify_params(2, args);
  temperature->setup();
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::update_langevin_coefficients()
{
  double dt = update->dt;
  double dt2 = 0.5 * dt;

  if (tstat_flag && damp_t > 0.0) {
    gamma_t = 1.0 / damp_t;
    lan_c1_t = exp(-gamma_t * dt);
    lan_c2_t = sqrt((1.0 - lan_c1_t * lan_c1_t) * boltz * t_target);
    lan_c1_t_2 = exp(-gamma_t * dt2);
    lan_c2_t_2 = sqrt((1.0 - lan_c1_t_2 * lan_c1_t_2) * boltz * t_target);
  }

  if (pstat_flag && damp_p > 0.0) {
    gamma_p = 1.0 / damp_p;
    lan_c1_p = exp(-gamma_p * dt);
    lan_c1_p_2 = exp(-gamma_p * dt2);

    double denom = (pstyle == ISO && pdim > 0) ? pdim : 1.0;  // shared iso kick uses variance / pdim
    lan_c2_p = sqrt((1.0 - lan_c1_p * lan_c1_p) * boltz * t_target / denom);
    lan_c2_p_2 = sqrt((1.0 - lan_c1_p_2 * lan_c1_p_2) * boltz * t_target / denom);
  }
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::integrate_temp_thermostat()
{
  if (!tstat_flag) return;
  compute_temp_target();
  update_langevin_coefficients();

  // nh_temp_flag selects Nose-Hoover chain; otherwise use the Langevin O-step.
  if (nh_temp_flag) nhc_temp_integrate();
  else langevin_temp();
}

/* ---------------------------------------------------------------------- */

void FixNHMiddle::integrate_press_thermostat()
{
  if (!pstat_flag || !mpchain) return;

  // nh_press_flag selects Nose-Hoover chain; otherwise use the Langevin O-step
  // on the barostat velocities.
  if (nh_press_flag) nhc_press_integrate();
  else langevin_press();
}

/* ----------------------------------------------------------------------
   first half of Verlet update in side ordering
------------------------------------------------------------------------- */

void FixNHMiddle::initial_integrate_side()
{
  integrate_press_thermostat();
  integrate_temp_thermostat();

  if (pstat_flag) {
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot_middle();
    nh_v_press();
  }

  nve_v();

  if (pstat_flag) remap();
  nve_x();

  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   second half of Verlet update in side ordering
------------------------------------------------------------------------- */

void FixNHMiddle::final_integrate_side()
{
  nve_v();

  if (pstat_flag) nh_v_press();

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot_middle();

  integrate_temp_thermostat();
  integrate_press_thermostat();
}

/* ----------------------------------------------------------------------
   first half of Verlet update in middle ordering
------------------------------------------------------------------------- */

void FixNHMiddle::initial_integrate(int /*vflag*/)
{
  if (integrator == SIDE) {
    initial_integrate_side();
    return;
  }

  // Middle ordering:
  //   B: velocity/barostat half updates around the force kick
  //   A: half position step
  //   O: thermostat/barostat stochastic or NH chain step
  //   A: second half position step
  if (pstat_flag) nh_v_press();
  nve_v();
  nve_v();
  if (pstat_flag) nh_v_press();

  if (pstat_flag) {
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot_middle();
  }

  if (pstat_flag) remap();
  nve_x_half();

  integrate_press_thermostat();
  integrate_temp_thermostat();

  nve_x_half();

  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   second half of Verlet update in middle ordering
------------------------------------------------------------------------- */

void FixNHMiddle::final_integrate()
{
  if (integrator == SIDE) {
    final_integrate_side();
    return;
  }

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot_middle();
}

/* ----------------------------------------------------------------------
   Langevin thermostat O-step on particle velocities
------------------------------------------------------------------------- */

void FixNHMiddle::langevin_temp()
{
  double lan_coeff1 = (integrator == MIDDLE) ? lan_c1_t : lan_c1_t_2;  // full O-step for middle, half O-step for side
  double lan_coeff2 = (integrator == MIDDLE) ? lan_c2_t : lan_c2_t_2;  // matching noise amplitude for the selected step size
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double mvv2e = force->mvv2e;
  int nlocal = atom->nlocal;
  double fsum[4] = {0.0, 0.0, 0.0, 0.0};
  double fsumall[4] = {0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < nlocal; i++) {
    double mass_i = rmass ? rmass[i] : mass[atom->type[i]];  // prefer per-atom mass when available
    double inv_sqrt_m = 1.0 / sqrt(mass_i * mvv2e);
    double kick[3] = {lan_coeff2 * random->gaussian() * inv_sqrt_m,
                      lan_coeff2 * random->gaussian() * inv_sqrt_m,
                      lan_coeff2 * random->gaussian() * inv_sqrt_m};

    if (zero_flag) {
      fsum[0] += mass_i * kick[0];
      fsum[1] += mass_i * kick[1];
      fsum[2] += mass_i * kick[2];
      fsum[3] += mass_i;
    }

    v[i][0] = lan_coeff1 * v[i][0] + kick[0];
    v[i][1] = lan_coeff1 * v[i][1] + kick[1];
    v[i][2] = lan_coeff1 * v[i][2] + kick[2];
  }

  if (zero_flag) {
    // Remove the mass-weighted center-of-mass component of the random kick so
    // the Langevin noise does not inject net linear momentum.
    MPI_Allreduce(fsum, fsumall, 4, MPI_DOUBLE, MPI_SUM, world);
    if (fsumall[3] > 0.0) {
      double correction[3] = {fsumall[0]/fsumall[3], fsumall[1]/fsumall[3], fsumall[2]/fsumall[3]};
      for (int i = 0; i < nlocal; i++) {
        v[i][0] -= correction[0];
        v[i][1] -= correction[1];
        v[i][2] -= correction[2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Langevin O-step on barostat velocities
------------------------------------------------------------------------- */

void FixNHMiddle::langevin_press()
{
  double lan_coeff1 = (integrator == MIDDLE) ? lan_c1_p : lan_c1_p_2;  // full O-step for middle, half O-step for side
  double lan_coeff2 = (integrator == MIDDLE) ? lan_c2_p : lan_c2_p_2;  // matching noise amplitude for omega_dot
  double kicks[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  if (comm->me == 0) {
    // Coupled pressure dimensions share the same random kick so the coupled
    // box lengths evolve consistently.
    if (pcouple == XYZ) {
      double kick = lan_coeff2 * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[1] = kicks[2] = kick;
    } else if (pcouple == XY) {
      double kick = lan_coeff2 * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[1] = kick;
      if (p_flag[2]) kicks[2] = lan_coeff2 * random->gaussian() / sqrt(omega_mass[2]);
    } else if (pcouple == YZ) {
      double kick = lan_coeff2 * random->gaussian() / sqrt(omega_mass[1]);
      kicks[1] = kicks[2] = kick;
      if (p_flag[0]) kicks[0] = lan_coeff2 * random->gaussian() / sqrt(omega_mass[0]);
    } else if (pcouple == XZ) {
      double kick = lan_coeff2 * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[2] = kick;
      if (p_flag[1]) kicks[1] = lan_coeff2 * random->gaussian() / sqrt(omega_mass[1]);
    } else {
      for (int i = 0; i < 6; i++)
        if (p_flag[i]) kicks[i] = lan_coeff2 * random->gaussian() / sqrt(omega_mass[i]);
    }
  }

  MPI_Bcast(kicks, 6, MPI_DOUBLE, 0, world);

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) omega_dot[i] = lan_coeff1 * omega_dot[i] + kicks[i];
}

/* ----------------------------------------------------------------------
   barostat force update matching fix_nh_new middle behavior
------------------------------------------------------------------------- */

void FixNHMiddle::nh_omega_dot_middle()
{
  double volume = (dimension == 3) ? domain->xprd*domain->yprd*domain->zprd : domain->xprd*domain->yprd;  // 3d volume, 2d area
  if (deviatoric_flag) compute_deviatoric();

  mtk_term1 = 0.0;
  if (mtk_flag) {
    if (pstyle == ISO) mtk_term1 = boltz * t_current;
    else {
      double *mvv_current = temperature->vector;
      for (int i = 0; i < 3; i++)
        if (p_flag[i]) mtk_term1 += mvv_current[i];
      mtk_term1 /= tdof;  // experimental
    }
  }

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      double f_omega = (p_current[i]-p_hydro)*volume / (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega*dthalf;
      omega_dot[i] *= pdrag_factor;
    }

  mtk_term2 = 0.0;
  if (mtk_flag) {
    for (int i = 0; i < 3; i++)
      if (p_flag[i]) mtk_term2 += omega_dot[i];
    if (tdof > 0.0) mtk_term2 /= tdof;  // experimental
  }

  if (pstyle == TRICLINIC) {
    for (int i = 3; i < 6; i++)
      if (p_flag[i]) {
        double f_omega = p_current[i]*volume/(omega_mass[i] * nktv2p);
        if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
        omega_dot[i] += f_omega*dthalf;
        omega_dot[i] *= pdrag_factor;
      }
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of positions
------------------------------------------------------------------------- */

void FixNHMiddle::nve_x_half()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dthalf * v[i][0];
      x[i][1] += dthalf * v[i][1];
      x[i][2] += dthalf * v[i][2];
    }
  }
}
