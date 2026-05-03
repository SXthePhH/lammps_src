#include "fix_npt_langevin_iso.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "modify.h"
#include "compute.h"
#include "update.h"
#include <cmath>
#include "kspace.h"
#include "random_mars.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{ISO,ANISO,TRICLINIC};
//constructor
FixNPTLangevin_iso::FixNPTLangevin_iso(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp,narg,arg),random(nullptr), id_temp(nullptr), id_press(nullptr)
{
    time_integrate = 1;
    dynamic_group_allow = 1;
    tcomputeflag = 1;
    pcomputeflag = 1;

    // some flags that I still don't know their functions, but I will add them
    // scalar_flag = 1;
    // vector_flag = 1;

    if (narg < 8) error->all(FLERR,"Illegal fix npt_langevin command");
    // there should be the gamma for thermo bath and baro bath
    // example: fix a all npt_langevin_iso temp press gamma_t gamma_p tau_baro seed
    int iarg = 3;
    while (iarg < narg){
        if (strcmp(arg[iarg], "temp") == 0){
            if (iarg+2 > narg) utils::missing_cmd_args(FLERR, fmt::format("fix {} temp", style), error);
            t_target = utils::numeric(FLERR,arg[iarg+1],false, lmp);
            damp_t = utils::numeric(FLERR,arg[iarg+2],false,lmp);
            iarg += 3;
        }
        else if (strcmp(arg[iarg], "iso") == 0){
            if (iarg+4 > narg) utils::missing_cmd_args(FLERR, fmt::format("fix {} iso", style), error);
            p_target = utils::numeric(FLERR,arg[iarg+1],false, lmp);
            pstyle = ISO;
            damp_p = utils::numeric(FLERR,arg[iarg+2],false,lmp);
            tau_baro = utils::numeric(FLERR,arg[iarg+3],false,lmp);
            omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
            p_freq = 1.0/tau_baro;
            iarg += 5;
        }
        else if (strcmp(arg[iarg], "aniso") == 0){   // this one is the same as the iso case
            if (iarg+4 > narg) utils::missing_cmd_args(FLERR, fmt::format("fix {} aniso", style), error);
            p_target = utils::numeric(FLERR,arg[iarg+1],false, lmp);
            pstyle = ANISO;
            damp_p = utils::numeric(FLERR,arg[iarg+2],false,lmp);
            tau_baro = utils::numeric(FLERR,arg[iarg+3],false,lmp);
            omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
            p_freq = 1.0/tau_baro;
            iarg += 5;
        }
        else if (strcmp(arg[iarg], "tri") == 0){   // this one is the same as the iso case
            if (iarg+4 > narg) utils::missing_cmd_args(FLERR, fmt::format("fix {} tri", style), error);
            p_target = utils::numeric(FLERR,arg[iarg+1],false, lmp);
            pstyle = TRICLINIC;
            damp_p = utils::numeric(FLERR,arg[iarg+2],false,lmp);
            tau_baro = utils::numeric(FLERR,arg[iarg+3],false,lmp);
            omega_mass_corr = utils::numeric(FLERR, arg[iarg+4], false, lmp);
            p_freq = 1.0/tau_baro;
            iarg += 5;
        }
        else {
            error->all(FLERR,"Illegal fix npt_langevin command");
        }
        // printf("iarg: %d", iarg);

    }




    // t_target = utils::numeric(FLERR,arg[iarg],false,lmp);
    // iarg++;
    // p_target = utils::numeric(FLERR,arg[iarg],false,lmp);
    // iarg++;
    // damp_t = utils::numeric(FLERR,arg[iarg],false,lmp);
    // iarg++;
    // damp_p = utils::numeric(FLERR,arg[iarg],false,lmp);
    // iarg++;
    // tau_baro = utils::numeric(FLERR,arg[iarg],false,lmp);  // compulsory
    // p_freq = 1.0 / tau_baro;
    // iarg++;  
    if (iarg < narg) {
        seed = utils::inumeric(FLERR,arg[iarg],false,lmp);
    } else {
        seed =
        123456789;
    }
    gamma_t = 1.0 / damp_t;
    gamma_p = 1.0 / damp_p;

    random = new RanMars(lmp,seed + comm->me);
    omega = 0.0;
    box_change |= BOX_CHANGE_X;
    box_change |= BOX_CHANGE_Y;
    box_change |= BOX_CHANGE_Z;
    restart_pbc = 1;
}

FixNPTLangevin_iso::~FixNPTLangevin_iso()
{
    if (tcomputeflag) modify->delete_compute(id_temp);
    if (pcomputeflag) modify->delete_compute(id_press);
}

int FixNPTLangevin_iso::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

int FixNPTLangevin_iso::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (tcomputeflag) {
      modify->delete_compute(id_temp);
      tcomputeflag = 0;
    }
    delete [] id_temp;
    id_temp = utils::strdup(arg[1]);

    int icompute = modify->find_compute(arg[1]);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,"Fix_modify temperature ID does not compute temperature");
    if (temperature->igroup != 0 && comm->me == 0)
      error->warning(FLERR,"Temperature for fix modify is not for group all");

    // reset id_temp of pressure correction to new ID

    if (pcomputeflag) {
      icompute = modify->find_compute(id_press);
      if (icompute < 0)
        error->all(FLERR,"Pressure ID for fix modify does not exist");
      modify->compute[icompute]->reset_extra_compute_fix(id_temp);
    }

    return 2;

  } else if (strcmp(arg[0],"press") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (pcomputeflag) {
      modify->delete_compute(id_press);
      pcomputeflag = 0;
    }
    delete [] id_press;
    id_press = utils::strdup(arg[1]);

    int icompute = modify->find_compute(arg[1]);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify pressure ID");
    pressure = modify->compute[icompute];

    if (pressure->pressflag == 0)
      error->all(FLERR,"Fix_modify pressure ID does not compute pressure");
    return 2;
  }

  return 0;
}


void FixNPTLangevin_iso::init()
{
    boltz = force->boltz;
    dt = update->dt;
    dt2 = dt * 0.5;
    dt4 = dt * 0.25;
    alpha = 1 + 1/atom->natoms ;
    lan_c1_t = exp(-gamma_t * dt);
    lan_c2_t = sqrt((1 - lan_c1_t * lan_c1_t) * boltz * t_target);

    lan_c1_p = exp(-gamma_p * dt);
    lan_c2_p = sqrt((1 - lan_c1_p * lan_c1_p) * boltz * t_target);
    omega_mass = omega_mass_corr * (3 * atom->natoms + 1) * boltz * t_target * tau_baro * tau_baro; 
    temperature = modify->get_compute_by_id(id_temp);
    pressure = modify->get_compute_by_id(id_press);
    if (force->kspace) kspace_flag = 1;
    else kspace_flag = 0;


}
// ...existing code...
void FixNPTLangevin_iso::setup(int /*vflag*/)
{
    int me = comm->me;
    long long step = update->ntimestep;
    int nlocal = atom->nlocal;

    // Temperature compute (safe to call in setup because Modify::init()/setup ordering)
    if (temperature) {
      t_current = temperature->compute_scalar();

    //   t_current = temperature->scalar;  // this seems redundant with compute_scalar()?
      if (me == 0) {
        // printf("DEBUG: temperature available t_current=%g dof=%d\n", t_current, temperature->dof);
        // fflush(stdout);
      }
    } else {
      if (me == 0) {
        // printf("DEBUG: no temperature compute (temperature == nullptr)\n");
        // fflush(stdout);
      }
    }

    if (pressure) {

        // pressure->compute_scalar(); //this seems redundant with compute_scalar()?
        // p_current = pressure->scalar;
        pressure->addstep(update->ntimestep+1); // ensure pressure compute has correct step count
        if (me == 0) {
        //   printf("DEBUG: pressure computed p_current=%g\n", p_current);
        //   fflush(stdout);
        }
      }
      else {
      if (me == 0) {
        // printf("DEBUG: no pressure compute (pressure == nullptr)\n");
        // fflush(stdout);
      }
     }

    double kt = boltz * t_target;
}
void FixNPTLangevin_iso::reset_dt()
{
    dt = update->dt;
    dt2 = dt * 0.5;
    dt4 = dt * 0.25;
}



double FixNPTLangevin_iso::fast_sinhc(double x)
{
    return 1 + (x*x)/6.0 + (x*x*x*x)/120.0 + (x*x*x*x*x*x)/5040.0 + (x*x*x*x*x*x*x*x)/362880.0 + (x*x*x*x*x*x*x*x*x*x)/39916800.0;
}

void FixNPTLangevin_iso::update_x()
{
    // update positions by full step
    double **x = atom->x;
    double factor2, factor4;
    factor4 = exp(omega * dt4);
    factor2 = factor4 * factor4;
    double Q_omega_dt4;

    Q_omega_dt4 = FixNPTLangevin_iso::fast_sinhc(omega * dt4);

    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
        if (i==0){
            double term1 = x[i][0] * factor2;
            double term2 = dt2 * (factor4 * atom->v[i][0] * Q_omega_dt4);
            // printf("x_check atom %d: term1=%g term2=%g\n", i, term1, term2);
            // printf("\n");
        }
        x[i][0] = x[i][0] * factor2 + dt2 * (factor4 * atom->v[i][0] * Q_omega_dt4);
        x[i][1] = x[i][1] * factor2 + dt2 * (factor4 * atom->v[i][1] * Q_omega_dt4);
        x[i][2] = x[i][2] * factor2 + dt2 * (factor4 * atom->v[i][2] * Q_omega_dt4);
    }
}


void FixNPTLangevin_iso::update_v()
{
    double **v = atom->v;
    double **f = atom->f;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int nlocal = atom->nlocal;
    double ftm2v = force->ftm2v;
    double dtf = 0.5 * dt * ftm2v;

    double factor2, factor4;
    factor4 = exp(omega * dt4 * alpha);
    if (update->ntimestep%1000 == 0 && comm->me == 0)     printf("factor4=%g\n", factor4);

    factor2 = exp(-omega * dt2 * alpha);
    double alpha_Q_omega_dt4;
    // printf("v_check: omega=%g dt4=%g\n factor4=%g factor2=%g\n", omega, dt4, factor4, factor2);
    // printf("\n");

    alpha_Q_omega_dt4 = FixNPTLangevin_iso::fast_sinhc(alpha * omega * dt4);
    if (rmass) {
        for (int i = 0; i < nlocal; i++) {

            if (i==0){
                double term1 = factor2 * v[i][0];
                double term2 = dtf * (factor4 * f[i][0]/rmass[i] * alpha_Q_omega_dt4);
                // printf("v_check atom %d: term1=%g term2=%g\n", i, term1, term2);
                // printf("\n");
            }

            v[i][0] = factor2 * v[i][0] + dtf * (factor4 * f[i][0]/rmass[i] * alpha_Q_omega_dt4);
            v[i][1] = factor2 * v[i][1] + dtf * (factor4 * f[i][1]/rmass[i] * alpha_Q_omega_dt4);
            v[i][2] = factor2 * v[i][2] + dtf * (factor4 * f[i][2]/rmass[i] * alpha_Q_omega_dt4);
        }
    } else {
        for (int i = 0; i < nlocal; i++) {
            if (i==0){
                double term1 = factor2 * v[i][0];
                double term2 = dtf * (factor4 * f[i][0]/mass[atom->type[i]] * alpha_Q_omega_dt4);
                // printf("v_check atom %d: term1=%g term2=%g\n", i, term1, term2);
                // printf("\n");
            }
            v[i][0] = factor2 * v[i][0] + dtf * (factor4 * f[i][0]/mass[atom->type[i]] * alpha_Q_omega_dt4);
            v[i][1] = factor2 * v[i][1] + dtf * (factor4 * f[i][1]/mass[atom->type[i]] * alpha_Q_omega_dt4);
            v[i][2] = factor2 * v[i][2] + dtf * (factor4 * f[i][2]/mass[atom->type[i]] * alpha_Q_omega_dt4);
        }
    }
}

void FixNPTLangevin_iso::langevin_press()
{
    double term2 = 0.0;

    // 1. 仅在 Rank 0 计算随机项
    if (comm->me == 0) {
        term2 = lan_c2_p * random->gaussian() / sqrt(omega_mass);
    }

    // 2. 将随机项广播给所有核
    MPI_Bcast(&term2, 1, MPI_DOUBLE, 0, world);

    // 3. 所有核执行相同的更新
    double term1 = lan_c1_p * omega;
    omega = term1 + term2;
}

void FixNPTLangevin_iso::langevin_temp()
{
    double **v = atom->v;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    double mvv2e = force->mvv2e;

    int nlocal = atom->nlocal;
    if (rmass) {
        for (int i = 0; i < nlocal; i++) {
            if (i==0){
                double term1 = lan_c1_t * v[i][0];
                double term2 = lan_c2_t * random->gaussian()/ sqrt(rmass[i]);
                // printf("temperature_check atom %d: term1=%g term2=%g\n", i, term1, term2);
                // printf("\n");
            }

            v[i][0] = lan_c1_t * v[i][0] + lan_c2_t * random->gaussian()/ sqrt(rmass[i] * mvv2e);
            v[i][1] = lan_c1_t * v[i][1] + lan_c2_t * random->gaussian()/ sqrt(rmass[i] * mvv2e);
            v[i][2] = lan_c1_t * v[i][2] + lan_c2_t * random->gaussian()/ sqrt(rmass[i] * mvv2e);
        }
    }

    else {
        for (int i = 0; i < nlocal; i++) {
            if (i==0){
                double term1 = lan_c1_t * v[i][0];
                double term2 = lan_c2_t * random->gaussian()/ sqrt(mass[atom->type[i]]);
                // printf("lan_c1_t=%g lan_c2_t=%g\n", lan_c1_t, lan_c2_t);
                // printf("velocity_check v[0][0]=%g\n", v[0][0]);
                // printf("temperature_check atom %d: term1=%g term2=%g\n", i, term1, term2);
                // printf("\n");
            }
            v[i][0] = lan_c1_t * v[i][0] + lan_c2_t * random->gaussian()/ sqrt(mass[atom->type[i]] * mvv2e);
            v[i][1] = lan_c1_t * v[i][1] + lan_c2_t * random->gaussian()/ sqrt(mass[atom->type[i]] * mvv2e);
            v[i][2] = lan_c1_t * v[i][2] + lan_c2_t * random->gaussian()/ sqrt(mass[atom->type[i]] * mvv2e);
        }
    }

}

void FixNPTLangevin_iso::update_omega()
{
    double volume;
    double nktv2p = force->nktv2p;
    volume = domain->xprd * domain->yprd * domain->zprd;
    // printf("omega_check: volume=%g p_current=%g p_target=%g t_current=%g\n", volume, p_current, p_target, t_current);
    double term1 = dt2*(3 * volume/omega_mass * (p_current -p_target)/nktv2p);
    double term2 = dt2*(3 * boltz * t_current/omega_mass);
    // printf("omega_check: term1=%g term2=%g\n", term1, term2);

    omega += dt2*(3 * volume/omega_mass * (p_current -p_target)/nktv2p + 3 * boltz * t_current/omega_mass);
}
// ...existing code...
void FixNPTLangevin_iso::remap()
{
    // isotropic linear scale factor
    // linear scale s = exp(omega * dt2)
    // volume scales as s^3 = exp(3 * omega * dt2)
    // check whether omega on different procs are the same, if not we may need to do an MPI_Allreduce here to get the global omega before calculating s, but for now we assume they are the same since they should be calculated from the same global pressure and temperature
    // get the step number
    int step = update->ntimestep;
    // printf("proc: %d remap_check: omega=%g step=%d\n", comm->me, omega, step);
    double s = exp(omega * dt2);

    // scale each box dimension about its center; do NOT change atom coordinates
    for (int d = 0; d < 3; ++d) {
      double oldlo = domain->boxlo[d];
      double oldhi = domain->boxhi[d];
      double center = 0.5 * (oldlo + oldhi);
      double half = 0.5 * (oldhi - oldlo);
      double new_half = half * s;
      domain->boxlo[d] = center - new_half;
      domain->boxhi[d] = center + new_half;
    }

    // update derived global/local box quantities
    domain->set_global_box();
    domain->set_local_box();

    // Intentionally do NOT call domain->x2lamda() / lamda2x() here:
    // atom->x (Cartesian) remain unchanged by this remap as requested.
    // If remapping causes atoms to be outside process subdomains you may need
    // to call irregular->migrate_atoms() (or other migration) at a safe point.
}
 // ...existing code...
void FixNPTLangevin_iso::initial_integrate(int /*vflag*/)
{
    // printf("initial_velocity: v[0][0]=%g\n", atom->v[0][0]);
    // printf("\n");

    int debug = 0;
    if (debug) omega = 0.0; // for testing, omega is always 0 to make it nvt, not npt
    update_v();
    temperature->compute_scalar();
    t_current = temperature->scalar;
    pressure->compute_scalar();
    p_current = pressure->scalar;
    pressure->addstep(update->ntimestep+1); // ensure pressure compute has correct step count

    update_omega();
    if (debug) omega = 0.0; // for testing, omega is always 0 to make it nvt, not npt
    remap();
    // force->kspace->setup(); // in case box changed, but this is not needed since we do not need to calculate force in next few steps
    update_x();
    langevin_temp();
    langevin_press();
    update_x();
    if (debug) omega = 0.0; // for testing, omega is always 0 to make it nvt, not npt

    // printf("initial_end_check: t_current=%g\n", t_current);
    // printf("\n");
    remap();
    if (kspace_flag) force->kspace->setup(); // in case box changed

}

void FixNPTLangevin_iso::final_integrate()
{
    int debug = 0;

    // printf("final_start_check: t_current=%g\n", t_current);
    // printf("\n");

    // // print out all the velocity
    // for (int i = 0; i < atom->nlocal; i++) {
    //     printf("final_integrate: atom %d v=%g %g %g\n", i, atom->v[i][0], atom->v[i][1], atom->v[i][2]);
    // }

    // printf("\n");

    t_current = temperature->compute_scalar();
    // printf("temperature_check: t_current=%g\n", t_current);
    // printf("\n");

    p_current = pressure->compute_scalar();
    pressure->addstep(update->ntimestep+1); // ensure pressure compute has correct step count

    update_omega();
    if (debug) omega = 0.0; // for testing, omega is always 0 to make it nvt, not npt
    update_v();

}
