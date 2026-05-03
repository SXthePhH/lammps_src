#ifdef FIX_CLASS
// clang-format off
FixStyle(npt_langevin_iso,FixNPTLangevin_iso);
// clang-format on
#else

#ifndef LMP_FIX_NPT_LANGEVIN_ISO_H
#define LMP_FIX_NPT_LANGEVIN_ISO_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNPTLangevin_iso : public Fix {
 public:
  FixNPTLangevin_iso(class LAMMPS *, int, char **);
  ~FixNPTLangevin_iso() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void reset_dt() override;
  int modify_param(int, char **) override;
  // void *extract(const char *, int &) override;
  // double compute_scalar() override;




 protected:
  // time and conversion
  double dt2, dt4, dt;
  // thermodynamic constants
  double boltz, nktv2p;
  // temperature
  double t_target, t_current, ke_target;
  int tstat_flag;
  // pressure / barostat (isotropic only)
  int pstat_flag;

  int pstyle; //ISO, ANISO, TRICLINIC


  double p_target;      // isotropic target pressure
  double p_current;
  double omega;         // scalar for isotropic barostat
  double omega_dot;

  double tau_baro;   // the relaxation time for barostat, this one is usually 1ps
  double p_freq; // barostat frequencies, p_freq = 1/tau_baro
  // omega_mass = (3N+1)* k_B * T * tau_baro^2
  double omega_mass_corr;
  double omega_mass;
  double vol0, t0;
//   double dt;            // store update->dt initially


//   int mpchain_default_flag;    // 1 = mpchain is default
  double gamma_t, gamma_p; // friction coefficients
  double damp_t, damp_p; // damping parameters
  double lan_c1_t, lan_c1_p, lan_c2_t, lan_c2_p; // Langevin coefficients
  double alpha;


  char *id_temp, *id_press;
  // pointers to computes
  class Compute *temperature;
  class Compute *pressure;
  int tcomputeflag, pcomputeflag;    // 1 = compute was created by fix
                                     // 0 = created externally
  int kspace_flag;

  class RanMars *random;
  // helper
  void couple();
  double fast_sinhc(double);
  void compute_temp_target();
  void compute_press_target();
  void update_v();
  void update_x();
  void update_omega();
  void remap();
  void langevin_temp();
  void langevin_press();
  
  int seed;

};

} // namespace LAMMPS_NS

#endif
#endif