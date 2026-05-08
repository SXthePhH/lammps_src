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
FixStyle(shake_new,FixShakeNew);
// clang-format on
#else

#ifndef LMP_FIX_SHAKE_NEW_H
#define LMP_FIX_SHAKE_NEW_H

#include "fix_shake.h"

namespace LAMMPS_NS {

class FixShakeNew : public FixShake {
 public:
  FixShakeNew(class LAMMPS *, int, char **);
  ~FixShakeNew() override;

  int setmask() override;
  void setup(int) override;
  void end_of_step() override;
  void reset_dt() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
