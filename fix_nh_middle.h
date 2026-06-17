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

#ifndef LMP_FIX_NH_MIDDLE_H
#define LMP_FIX_NH_MIDDLE_H

#include "fix_nh.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixNHMiddle : public FixNH {
 public:
  FixNHMiddle(class LAMMPS *, int, char **);
  ~FixNHMiddle() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void reset_dt() override;

 protected:
  struct ArgList {
    std::vector<std::string> storage;
    std::vector<char *> argv;
  };

  static ArgList filter_middle_args(int, char **);
  FixNHMiddle(class LAMMPS *, int, char **, ArgList &&);

  void nve_x_half();
  void initial_integrate_side();
  void final_integrate_side();
  void integrate_temp_thermostat();
  void integrate_press_thermostat();
  void langevin_temp();
  void langevin_press();
  void nh_omega_dot_middle();
  void parse_middle_args(int, char **);
  void update_langevin_coefficients();
  void update_middle_barostat_masses();

  double gamma_t, gamma_p, damp_t, damp_p;
  double lan_c1_t, lan_c2_t;
  double lan_c1_t_2, lan_c2_t_2;
  double lan_c1_p, lan_c2_p;
  double lan_c1_p_2, lan_c2_p_2;
  double omega_mass_corr, tau_baro;
  class RanMars *random;
  int seed, zero_flag;
  int integrator;
  int nh_temp_flag, nh_press_flag;
  int big_mass_flag, big_omega_update_flag;
};

}    // namespace LAMMPS_NS

#endif
