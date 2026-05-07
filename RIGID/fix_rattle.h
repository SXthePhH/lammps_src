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
FixStyle(rattle,FixRattle);
// clang-format on
#else

#ifndef LMP_FIX_RATTLE_H
#define LMP_FIX_RATTLE_H

#include "fix_shake.h"

namespace LAMMPS_NS {

class FixRattle : public FixShake {
 public:
  double derr_max;    // distance error
  double verr_max;    // velocity error

  FixRattle(class LAMMPS *, int, char **);
  ~FixRattle() override;

  // extra protected constructor for wrapper classes
 protected:
  FixRattle(class LAMMPS *, int, char **, int);

 public:
  int setmask() override;
  void init() override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void final_integrate() override;
  void final_integrate_respa(int, int) override;

  void correct_coordinates(int vflag) override;
  void shake_end_of_step(int vflag) override;
  void end_of_step() override;

  void reset_dt() override;

 protected:
  void update_v_half_nocons();
  void update_v_half_nocons_respa(int);

  // debugging methods

  bool check3angle(double **v, int m, bool checkr, bool checkv);
  bool check2(double **v, int m, bool checkr, bool checkv);
  bool check3(double **v, int m, bool checkr, bool checkv);
  bool check4(double **v, int m, bool checkr, bool checkv);
  bool check_constraints(double **v, bool checkr, bool checkv);
};

}    // namespace LAMMPS_NS

#endif
#endif
