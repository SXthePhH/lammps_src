/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nhcon,FixNHCon);
// clang-format on
#else

#ifndef LMP_FIX_NHCON_H
#define LMP_FIX_NHCON_H

#include "fix_rattle.h"

namespace LAMMPS_NS {

class FixNHCon : public FixRattle {
 public:
  FixNHCon(class LAMMPS *, int, char **);
  ~FixNHCon() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void initial_integrate_respa(int, int, int) override;
  void pre_force_respa(int, int, int) override;
  void final_integrate_respa(int, int) override;
  void pre_exchange() override;
  double compute_scalar() override;
  double compute_vector(int) override;
  std::string get_thermo_colname(int) override;
  void write_restart(FILE *) override;
  virtual int pack_restart_data(double *);
  void restart(char *) override;
  int modify_param(int, char **) override;
  void reset_target(double) override;
  void reset_dt() override;
  void *extract(const char *, int &) override;
  double memory_usage() override;

  void initial_integrate_middle();
  void final_integrate_middle();

 protected:
  // constraint helpers -- call FixRattle internals at integration points
  void apply_rattle_velocity_constraints();
  void apply_shake_position_constraints(int vflag);

  int dimension, which;
  double dtf, dthalf, dt4, dt8, dto;
  double boltz, nktv2p, tdof;
  double vol0;
  double t0;
  double t_start, t_stop;
  double t_current, t_target, ke_target;
  double t_freq;

  int tstat_flag;
  int pstat_flag;

  int pstyle, pcouple, allremap;
  int p_flag[6];
  double p_start[6], p_stop[6];
  double p_freq[6], p_target[6];
  double omega[6], omega_dot[6];
  double omega_mass[6];
  double p_current[6];
  double drag, tdrag_factor;
  double pdrag_factor;
  int kspace_flag;
  int dilate_group_bit;
  std::vector<Fix *> rfix;
  char *id_dilate;
  class Irregular *irregular;

  double p_temp;
  int p_temp_flag;

  char *id_temp, *id_press;
  class Compute *temperature, *pressure;
  int tcomputeflag, pcomputeflag;

  double *eta, *eta_dot;
  double *eta_dotdot;
  double *eta_mass;
  int mtchain;
  int mtchain_default_flag;

  double *etap;
  double *etap_dot;
  double *etap_dotdot;
  double *etap_mass;
  int mpchain;

  int mtk_flag;
  int pdim;
  double p_freq_max;

  double p_hydro;

  int nc_tchain, nc_pchain;
  double factor_eta;
  double sigma[6];
  double fdev[6];
  int deviatoric_flag;
  double h0_inv[6];
  int nreset_h0;

  double mtk_term1, mtk_term2;

  int eta_mass_flag;
  int omega_mass_flag;
  int etap_mass_flag;
  int dipole_flag;
  int dlm_flag;
  int integrator;

  int scaleyz;
  int scalexz;
  int scalexy;
  int flipflag;

  int pre_exchange_flag;

  double fixedpoint[3];

  double gamma_t, gamma_p, damp_t, damp_p;
  double lan_c1_t, lan_c2_t;
  double lan_c1_t_2, lan_c2_t_2;
  double lan_c1_p, lan_c2_p;
  double lan_c1_p_2, lan_c2_p_2;
  int zero_flag;

  double omega_mass_corr, tau_baro;
  class RanMars *random;
  int seed;
  double **v_storage;
  double **v_backup;
  int v_stored;
  int nmax;

  void couple();
  virtual void remap();
  void nhc_temp_integrate();
  void nhc_press_integrate();

  virtual void nve_x();
  virtual void nve_x_half();
  virtual void nve_v();
  virtual void nh_v_press();
  virtual void nh_v_temp();
  virtual void compute_temp_target();
  virtual int size_restart_global();

  void compute_sigma();
  void compute_deviatoric();
  double compute_strain_energy();
  void compute_press_target();
  void nh_omega_dot();
  void langevin_temp();
  void langevin_press();

  double force_omega[6];
  void compute_f_omega();
  void update_omega_dot();
  double omegadot_exp[6];
  void mat_exp(double delta_t, const double *mat);
  void remap_me();
  void scale_v();
  void nhc_press_integrate_me();
  void nhc_press_integrate_iso();
  void end_of_step() override;
  int nh_temp_flag, nh_press_flag;
  int big_mass_flag, big_omega_update_flag;
};

}    // namespace LAMMPS_NS

#endif
#endif
