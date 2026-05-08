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

#include "fix_shake_new.h"

#include "atom.h"
#include "force.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{V,VP,XSHAKE};

/* ---------------------------------------------------------------------- */

FixShakeNew::FixShakeNew(LAMMPS *lmp, int narg, char **arg) :
    FixShake(lmp, narg, arg)
{
  rattle = 1;

  // FixShake::grow_arrays was called in base constructor with rattle=0,
  // so call again with rattle=1 to allocate vp

  vp = nullptr;
  FixShake::grow_arrays(atom->nmax);

  comm_mode = XSHAKE;
}

/* ---------------------------------------------------------------------- */

FixShakeNew::~FixShakeNew()
{
  // vp is destroyed by FixShake::~FixShake
}

/* ---------------------------------------------------------------------- */

int FixShakeNew::setmask()
{
  int mask = 0;
  mask |= PRE_NEIGHBOR;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_PRE_REVERSE;
  mask |= MIN_POST_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

/* ----------------------------------------------------------------------
   call parent setup, then fix dtfsq/dt_inner to SHAKE values
   (rattle=1 causes half-step dtfsq, but SHAKE position constraint
    in post_force needs full-step dtfsq)
------------------------------------------------------------------------- */

void FixShakeNew::setup(int vflag)
{
  FixShake::setup(vflag);

  if (!respa)
    dtfsq = update->dt * update->dt * force->ftm2v;
  else
    dtf_inner = step_respa[0] * force->ftm2v;
}

/* ----------------------------------------------------------------------
   RATTLE-style velocity correction at end of timestep:
   remove velocity components along constrained bonds
------------------------------------------------------------------------- */

void FixShakeNew::end_of_step()
{
  correct_velocities();
}

/* ----------------------------------------------------------------------
   fix dtfsq/dt_inner to SHAKE values after parent reset_dt
------------------------------------------------------------------------- */

void FixShakeNew::reset_dt()
{
  FixShake::reset_dt();

  if (utils::strmatch(update->integrate_style,"^verlet"))
    dtfsq = update->dt * update->dt * force->ftm2v;
  else
    dtf_inner = step_respa[0] * force->ftm2v;
}
