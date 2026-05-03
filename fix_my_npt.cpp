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

#include "fix_my_npt.h"

#include "error.h"
#include "modify.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMYNPT::FixMYNPT(LAMMPS *lmp, int narg, char **arg) : FixNPTLangevin_iso(lmp, narg, arg)
{
  // create a new compute temp style
  // id = fix-ID + temp
  // compute group = all since pressure is always global (group all)
  // and thus its KE/temperature contribution should use group all

  id_temp = utils::strdup(std::string(id) + "_temp");

  // printf("id_temp: %s\n", id_temp);
  
  modify->add_compute(fmt::format("{} all temp", id_temp));
  tcomputeflag = 1;

  // create a new compute pressure style
  // id = fix-ID + press, compute group = all
  // pass id_temp as 4th arg to pressure constructor

  id_press = utils::strdup(std::string(id) + "_press");

  // printf("id_press: %s\n", id_press);

  modify->add_compute(fmt::format("{} all pressure {}", id_press, id_temp));
  pcomputeflag = 1;
}
