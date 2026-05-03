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

/* ----------------------------------------------------------------------
   Drop-in header for fix_nh_langevin: replaces Nose-Hoover chains with
   Langevin stochastic dynamics while keeping the same public/protected
   interface as fix_nh so that all existing child classes
   (fix_nvt, fix_npt, fix_nph, fix_nvt/sphere, fix_nvt/sllod, …)
   continue to compile and link correctly.

   Differences from fix_nh.h:
     - NHC chain arrays (eta*, etap*, mtchain, mpchain, nc_t/pchain)
       removed; replaced by Langevin fields
     - langevin_temp() / langevin_press() replace nhc_temp/press_integrate()
     - All virtual method signatures are preserved verbatim from fix_nh.h
     - Fields kept for subclass ABI compatibility are marked accordingly
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nh/langevin,FixNHLangevin)
// clang-format on
#else

#ifndef LMP_FIX_NH_LANGEVIN_H
#define LMP_FIX_NH_LANGEVIN_H

#include "fix.h"    // IWYU pragma: export

namespace LAMMPS_NS {

class FixNHLangevin : public Fix {
 public:
  FixNHLangevin(class LAMMPS *, int, char **);
  ~FixNHLangevin() override;
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
  void write_restart(FILE *) override;
  virtual int pack_restart_data(double *);    // pack restart data
  void restart(char *) override;
  int modify_param(int, char **) override;
  void reset_target(double) override;
  void reset_dt() override;
  void *extract(const char *, int &) override;
  double memory_usage() override;

 protected:
  int dimension, which;
  double dtv, dtf, dthalf, dt4, dt8, dto;
  double boltz, nktv2p, tdof;    // tdof is double, matching fix_nh.h
  double vol0;    // reference volume
  double t0;      // reference temperature (used for barostat mass)
  double t_start, t_stop;
  double t_current, t_target, ke_target;
  double t_freq;    // kept for interface compat; not used in Langevin logic

  int tstat_flag;    // 1 if control T
  int pstat_flag;    // 1 if control P

  int pstyle, pcouple, allremap;
  int p_flag[6];    // 1 if control P on this dim, 0 if not
  double p_start[6], p_stop[6];
  double p_freq[6], p_target[6];
  double omega[6], omega_dot[6];
  double omega_mass[6];
  double p_current[6];
  double drag, tdrag_factor;     // drag factor on particle thermostat
  double pdrag_factor;           // drag factor on barostat
  int kspace_flag;               // 1 if KSpace invoked, 0 if not
  int dilate_group_bit;          // mask for dilation group
  std::vector<Fix *> rfix;       // list of rigid fixes
  char *id_dilate;               // group name to dilate
  class Irregular *irregular;    // for migrating atoms after box flips

  double p_temp;    // target temperature for barostat
  int p_temp_flag;

  int nlevels_respa;
  double *step_respa;

  char *id_temp, *id_press;
  class Compute *temperature, *pressure;
  int tcomputeflag, pcomputeflag;    // 1 = compute was created by fix
                                     // 0 = created externally

  // NHC chain arrays removed; the following stubs are kept so that
  // child classes that reference mtchain_default_flag still compile.
  int mtchain_default_flag;    // always 1; read by FixNVTSllod

  int mtk_flag;         // 1 for MTK correction (always on)
  int pdim;             // number of barostatted dims
  double p_freq_max;    // maximum barostat frequency (= 1/damp_p)

  double p_hydro;    // hydrostatic target pressure

  // nc_tchain / nc_pchain removed (no NHC loops)

  double factor_eta;    // kept as virtual-method datum for nh_v_temp / subclasses
  double sigma[6];        // scaled target stress
  double fdev[6];         // deviatoric force on barostat
  int deviatoric_flag;    // 0 if target stress tensor is hydrostatic
  double h0_inv[6];       // h_inv of reference (zero strain) box
  int nreset_h0;          // interval for resetting h0

  double mtk_term1, mtk_term2;    // Martyna-Tobias-Klein corrections

  int eta_mass_flag;      // unused in Langevin; kept for subclass interface
  int omega_mass_flag;    // 1 if omega_mass updated, 0 if not
  int etap_mass_flag;     // unused in Langevin; kept for subclass interface
  int dipole_flag;        // 1 if dipole is updated, 0 if not
  int dlm_flag;           // 1 if using the DLM rotational integrator, 0 if not

  int scaleyz;     // 1 if yz scaled with lz
  int scalexz;     // 1 if xz scaled with lz
  int scalexy;     // 1 if xy scaled with ly
  int flipflag;    // 1 if box flips are invoked as needed

  int pre_exchange_flag;    // set if pre_exchange needed for box flips

  double fixedpoint[3];    // location of dilation fixed-point

  // ---- Langevin-specific fields ----
  double damp_t;          // thermostat Langevin damping time
  double damp_p;          // barostat  Langevin damping time
  int    lan_seed;        // RNG seed
  double gamma_t, gamma_p;       // friction coefficients (= 1/damp)
  double lan_c1_t, lan_c2_t;    // Ornstein-Uhlenbeck coefficients, thermostat
  double lan_c1_p, lan_c2_p;    // Ornstein-Uhlenbeck coefficients, barostat
  class RanMars *random;         // Marsaglia RNG

  // ---- Langevin integration kernels (replace NHC routines) ----
  void langevin_temp();    // O-step on particle velocities
  void langevin_press();   // O-step on barostat velocities (omega_dot)

  // ---- Helper methods — signatures identical to fix_nh.h ----
  void couple();
  virtual void remap();
  virtual void nve_x();           // may be overwritten by child classes
  virtual void nve_v();
  virtual void nh_v_press();
  virtual void nh_v_temp();       // no-op when tstat_flag=0; virtual for subclasses
  virtual void compute_temp_target();
  virtual int size_restart_global();

  void compute_sigma();
  void compute_deviatoric();
  double compute_strain_energy();
  void compute_press_target();
  void nh_omega_dot();
};

}    // namespace LAMMPS_NS

#endif
#endif
