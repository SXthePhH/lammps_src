#ifdef FIX_CLASS
// clang-format off
FixStyle(npt_langevin,FixNPTLangevin);
// clang-format on
#else

#ifndef LMP_FIX_NPT_LANGEVIN_H
#define LMP_FIX_NPT_LANGEVIN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNPTLangevin : public Fix {
 public:
  FixNPTLangevin(class LAMMPS *, int, char **);
  ~FixNPTLangevin() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void reset_dt() override;
  int modify_param(int, char **) override; // 添加这一行

 protected:
  int dimension;

  // time
  double dt, dt2, dt4;

  // thermodynamic constants
  double boltz,tdof; // tdof is the thermostat degrees of freedom

  // thermostat
  double t_target, t_current;
  double damp_t, gamma_t;
  double lan_c1_t, lan_c2_t;

  // MTK correction: alpha = 1 + 1/N
  double alpha;

  // pstyle: 0=ISO, 1=ANISO, 2=TRICLINIC
  // pcouple: 0=NONE, 1=XYZ, 2=XY, 3=YZ, 4=XZ
  int pstyle, pcouple;

  // 6-component barostat in Voigt order: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
  // (same convention as fix_nh)
  int    p_flag[6];       // 1 if this component is barostatted
  double p_target[6];     // target stress per component
  double p_current[6];    // current stress per component
  double omega[6];        // barostat "velocity" per component
  double omega_exp[6];
  double omega_mass[6];   // barostat mass per component
  double omega_mass_corr; // user scaling factor for barostat mass
  double p_hydro;         // hydrostatic target (average of active diagonal)

  double damp_p, gamma_p;
  double lan_c1_p, lan_c2_p;
  double lan_c1_p_2, lan_c2_p_2; // precompute exp(-gamma_p*dt/2) and corresponding c2 for kick-drift-kick
  double tau_baro;        // relaxation time (same for all components)

  // tilt-component scaling flags (mirrors fix_nh logic)
  int scalexy, scalexz, scaleyz;

  int kspace_flag;

  char *id_temp, *id_press;
  class Compute *temperature;
  class Compute *pressure;
  int tcomputeflag, pcomputeflag;

  class RanMars *random;
  int seed;

  // helpers
  double fast_sinhc(double x);
  void couple();
  void update_v();
  void scale_v();
  void update_x();
  void scale_x();
  void update_omega();
  void remap();
  void langevin_temp();
  void langevin_press();
  void mat_exp(double delta_t);

  double compute_temp_manual(); // for testing only, compute temperature from velocities without thermostat correction

 protected:
  double **v_storage; // Stores v(t+dt) to restore at next initial_integrate
  double **v_backup;  // Stores v(t+dt/2) captured at start of final_integrate
  int v_stored;
  int nmax;

  // override to perform the "thermo hack"
  void end_of_step() override;
};

} // namespace LAMMPS_NS

#endif
#endif