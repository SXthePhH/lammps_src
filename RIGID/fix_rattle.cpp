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

/* ----------------------------------------------------------------------
   Contributing author: Peter Wirnsberger (University of Cambridge)
------------------------------------------------------------------------- */

#include "fix_rattle.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathExtra;

// set RATTLE_DEBUG  1 to check constraints at end of timestep

#define RATTLE_DEBUG 0

// set RATTLE_RAISE_ERROR 1 if you want this fix to raise
//     an error if the constraints cannot be satisfied

#define RATTLE_RAISE_ERROR 0

// You can enable velocity and coordinate checks separately
// by setting RATTLE_TEST_VEL/POS true

#define RATTLE_TEST_VEL false
#define RATTLE_TEST_POS false

enum{V,VP,XSHAKE};

/* ---------------------------------------------------------------------- */

FixRattle::FixRattle(LAMMPS *lmp, int narg, char **arg) :
  FixShake(lmp, narg, arg)
{
  rattle = 1;

  // allocate memory for unconstrained velocity update
  // FixShake::grow_arrays was called in base constructor with rattle=0,
  // so call again with rattle=1 to allocate vp

  vp = nullptr;
  FixShake::grow_arrays(atom->nmax);

  comm_mode = XSHAKE;

  verr_max = 0;
  derr_max = 0;
}

/* ----------------------------------------------------------------------
   Protected constructor for wrapper classes (e.g. fix nh new).
   Calls the no-arg-parsing FixShake constructor.
------------------------------------------------------------------------- */

FixRattle::FixRattle(LAMMPS *lmp, int narg, char **arg, int dummy) :
    FixShake(lmp, narg, arg, dummy)
{
  rattle = 1;

  vp = nullptr;
  FixShake::grow_arrays(atom->nmax);

  comm_mode = XSHAKE;

  verr_max = 0;
  derr_max = 0;
}

/* ---------------------------------------------------------------------- */

FixRattle::~FixRattle()
{
  // vp is now destroyed by FixShake::~FixShake

#if RATTLE_DEBUG

  // communicate maximum distance error

  double global_derr_max, global_verr_max;
  int npid;

  MPI_Reduce(&derr_max, &global_derr_max, 1 , MPI_DOUBLE, MPI_MAX, 0, world);
  MPI_Reduce(&verr_max, &global_verr_max, 1 , MPI_DOUBLE, MPI_MAX, 0, world);

  if (comm->me == 0) {
    utils::logmesg(lmp, "RATTLE: Maximum overall relative position error ( (r_ij-d_ij)/d_ij ): "
                   "{:.10}\n", global_derr_max);
    utils::logmesg(lmp, "RATTLE: Maximum overall absolute velocity error (r_ij * v_ij): {:.10}\n",
                   global_verr_max);
  }
#endif
}

/* ---------------------------------------------------------------------- */

int FixRattle::setmask()
{
  int mask = 0;
  mask |= PRE_NEIGHBOR;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= FINAL_INTEGRATE;
  mask |= FINAL_INTEGRATE_RESPA;
  mask |= MIN_POST_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

/* ----------------------------------------------------------------------
   initialize RATTLE and check that this is the last final_integrate fix
------------------------------------------------------------------------- */

void FixRattle::init() {

  // initialize SHAKE first

  FixShake::init();

  // show a warning if any final-integrate fix comes after this one

  int after = 0;
  int flag = 0;
  for (int i = 0; i < modify->nfix; i++) {
    if (strcmp(id,modify->fix[i]->id) == 0) after = 1;
    else if ((modify->fmask[i] & FINAL_INTEGRATE) && after) flag = 1;
  }

  if (flag && comm->me == 0)
    error->warning(FLERR, "Fix rattle should come after all other integration fixes ");
}

/* ----------------------------------------------------------------------
   This method carries out an unconstrained velocity update first and
   then applies the velocity corrections directly (v and vp are modified).
------------------------------------------------------------------------- */

void FixRattle::post_force(int vflag)
{
  // remember vflag for the coordinate correction in this->final_integrate

  vflag_post_force = vflag;

  // unconstrained velocity update by half a timestep
  // similar to FixShake::unconstrained_update()

  update_v_half_nocons();

  // communicate the unconstrained velocities

  if (comm->nprocs > 1) {
    comm_mode = VP;
    comm->forward_comm(this);
  }

  // correct the velocity for each molecule accordingly

  int m;
  for (int i = 0; i < nlist; i++) {
    m = list[i];
    if      (shake_flag[m] == 2)        vrattle2(m);
    else if (shake_flag[m] == 3)        vrattle3(m);
    else if (shake_flag[m] == 4)        vrattle4(m);
    else                                vrattle3angle(m);
  }
}

/* ---------------------------------------------------------------------- */

void FixRattle::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  // remember vflag for the coordinate correction in this->final_integrate

  vflag_post_force = vflag;

  // unconstrained velocity update by half a timestep
  // similar to FixShake::unconstrained_update()

  update_v_half_nocons_respa(ilevel);

  // communicate the unconstrained velocities

  if (comm->nprocs > 1) {
    comm_mode = VP;
    comm->forward_comm(this);
  }

  // correct the velocity for each molecule accordingly

  int m;
  for (int i = 0; i < nlist; i++) {
    m = list[i];
    if      (shake_flag[m] == 2)        vrattle2(m);
    else if (shake_flag[m] == 3)        vrattle3(m);
    else if (shake_flag[m] == 4)        vrattle4(m);
    else                                vrattle3angle(m);
  }
}

/* ----------------------------------------------------------------------
   let SHAKE calculate the constraining forces for the coordinates
------------------------------------------------------------------------- */

void FixRattle::final_integrate()
{
  comm_mode = XSHAKE;
  FixShake::post_force(vflag_post_force);
}

/* ---------------------------------------------------------------------- */

void FixRattle::final_integrate_respa(int ilevel, int iloop)
{
  comm_mode = XSHAKE;
  FixShake::post_force_respa(vflag_post_force, ilevel, iloop);
}

/* ---------------------------------------------------------------------- */

void FixRattle::reset_dt()
{
  FixShake::reset_dt();
}

/* ----------------------------------------------------------------------
   carry out an unconstrained velocity update (vp is modified)
------------------------------------------------------------------------- */

void FixRattle::update_v_half_nocons()
{
  const double dtfv = 0.5 * update->dt * force->ftm2v;
  double dtfvinvm;

  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      if (shake_flag[i]) {
        dtfvinvm = dtfv / rmass[i];
        for (int k=0; k<3; k++)
          vp[i][k] = v[i][k] + dtfvinvm * f[i][k];
      }
      else
        vp[i][0] = vp[i][1] = vp[i][2] = 0;
    }
  }
  else {
    for (int i = 0; i < nlocal; i++) {
      dtfvinvm = dtfv / mass[type[i]];
      if (shake_flag[i]) {
        for (int k=0; k<3; k++)
          vp[i][k] = v[i][k] + dtfvinvm * f[i][k];
      }
      else
        vp[i][0] = vp[i][1] = vp[i][2] = 0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRattle::update_v_half_nocons_respa(int /*ilevel*/)
{
  // carry out unconstrained velocity update

  update_v_half_nocons();
}

/* ----------------------------------------------------------------------
  Let shake calculate new constraining forces for the coordinates;
  As opposed to the regular shake call, this method is usually called from
  end_of_step fixes after the second velocity integration has happened.
------------------------------------------------------------------------- */

void FixRattle::shake_end_of_step(int vflag) {

  if (comm->nprocs > 1) {
    comm_mode = V;
    comm->forward_comm(this);
  }

  comm_mode = XSHAKE;
  FixShake::shake_end_of_step(vflag);
}


/* ----------------------------------------------------------------------
  Let shake calculate new constraining forces and correct the
  coordinates. Nothing to do for rattle here.
------------------------------------------------------------------------- */

void FixRattle::correct_coordinates(int vflag) {
  comm_mode = XSHAKE;
  FixShake::correct_coordinates(vflag);
}

/* ----------------------------------------------------------------------
   Apply RATTLE velocity constraint at end of timestep:
   remove velocity components along constrained bonds.
   Debug checks are performed if RATTLE_DEBUG / RATTLE_RAISE_ERROR are on.
------------------------------------------------------------------------- */

void FixRattle::end_of_step()
{
  // apply velocity constraint
  correct_velocities();

  // debug checks
#if RATTLE_RAISE_ERROR
  if (comm->nprocs > 1) {
    comm_mode = V;
    comm->forward_comm(this);
  }
  if (!check_constraints(v, RATTLE_TEST_POS, RATTLE_TEST_VEL))
    error->one(FLERR, "Rattle failed ");
#endif
}

/* ---------------------------------------------------------------------- */

bool FixRattle::check_constraints(double **v, bool checkr, bool checkv)
{
  int m;
  bool ret = true;
  int i=0;
  while (i < nlist && ret) {
    m = list[i];
    if      (shake_flag[m] == 2)     ret = check2(v,m,checkr,checkv);
    else if (shake_flag[m] == 3)     ret = check3(v,m,checkr,checkv);
    else if (shake_flag[m] == 4)     ret = check4(v,m,checkr,checkv);
    else                             ret = check3angle(v,m,checkr,checkv);
    i++;
  }
  return ret;
}

/* ---------------------------------------------------------------------- */

bool FixRattle::check2(double **v, int m, bool checkr, bool checkv)
{
  bool      stat;
  double    r01[3],v01[3];
  const double tol = tolerance;
  double bond1 = bond_distance[shake_type[m][0]];

  tagint i0 = atom->map(shake_atom[m][0]);
  tagint i1 = atom->map(shake_atom[m][1]);

  MathExtra::sub3(x[i1],x[i0],r01);
  domain->minimum_image(FLERR, r01);
  MathExtra::sub3(v[i1],v[i0],v01);

  stat = !checkr || (fabs(sqrt(MathExtra::dot3(r01,r01)) - bond1) <= tol);
  if (!stat) error->one(FLERR,"Coordinate constraints are not satisfied up to desired tolerance ");

  stat = !checkv || (fabs(MathExtra::dot3(r01,v01)) <= tol);
  if (!stat) error->one(FLERR,"Velocity constraints are not satisfied up to desired tolerance ");
  return stat;
}

/* ---------------------------------------------------------------------- */

bool FixRattle::check3(double **v, int m, bool checkr, bool checkv)
{
  bool      stat;
  tagint    i0,i1,i2;
  double    r01[3], r02[3], v01[3], v02[3];
  const double tol = tolerance;
  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];

  i0 = atom->map(shake_atom[m][0]);
  i1 = atom->map(shake_atom[m][1]);
  i2 = atom->map(shake_atom[m][2]);

  MathExtra::sub3(x[i1],x[i0],r01);
  MathExtra::sub3(x[i2],x[i0],r02);

  domain->minimum_image(FLERR, r01);
  domain->minimum_image(FLERR, r02);

  MathExtra::sub3(v[i1],v[i0],v01);
  MathExtra::sub3(v[i2],v[i0],v02);

  stat = !checkr || (fabs(sqrt(MathExtra::dot3(r01,r01)) - bond1) <= tol &&
                      fabs(sqrt(MathExtra::dot3(r02,r02))-bond2) <= tol);
  if (!stat) error->one(FLERR,"Coordinate constraints are not satisfied up to desired tolerance ");

  stat = !checkv || (fabs(MathExtra::dot3(r01,v01)) <= tol &&
                      fabs(MathExtra::dot3(r02,v02)) <= tol);
  if (!stat) error->one(FLERR,"Velocity constraints are not satisfied up to desired tolerance ");
  return stat;
}

/* ---------------------------------------------------------------------- */

bool FixRattle::check4(double **v, int m, bool checkr, bool checkv)
{
  bool stat = true;
  const double tol = tolerance;
  double r01[3], r02[3], r03[3], v01[3], v02[3], v03[3];

  int i0 = atom->map(shake_atom[m][0]);
  int i1 = atom->map(shake_atom[m][1]);
  int i2 = atom->map(shake_atom[m][2]);
  int i3 = atom->map(shake_atom[m][3]);
  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];
  double bond3 = bond_distance[shake_type[m][2]];

  MathExtra::sub3(x[i1],x[i0],r01);
  MathExtra::sub3(x[i2],x[i0],r02);
  MathExtra::sub3(x[i3],x[i0],r03);

  domain->minimum_image(FLERR, r01);
  domain->minimum_image(FLERR, r02);
  domain->minimum_image(FLERR, r03);

  MathExtra::sub3(v[i1],v[i0],v01);
  MathExtra::sub3(v[i2],v[i0],v02);
  MathExtra::sub3(v[i3],v[i0],v03);

  stat = !checkr || (fabs(sqrt(MathExtra::dot3(r01,r01)) - bond1) <= tol &&
                      fabs(sqrt(MathExtra::dot3(r02,r02))-bond2) <= tol &&
                      fabs(sqrt(MathExtra::dot3(r03,r03))-bond3) <= tol);
  if (!stat) error->one(FLERR,"Coordinate constraints are not satisfied up to desired tolerance ");

  stat = !checkv || (fabs(MathExtra::dot3(r01,v01)) <= tol &&
                      fabs(MathExtra::dot3(r02,v02)) <= tol &&
                      fabs(MathExtra::dot3(r03,v03)) <= tol);
  if (!stat) error->one(FLERR,"Velocity constraints are not satisfied up to desired tolerance ");
  return stat;
}

/* ---------------------------------------------------------------------- */

bool FixRattle::check3angle(double **v, int m, bool checkr, bool checkv)
{
  bool stat = true;
  const double tol = tolerance;
  double r01[3], r02[3], r12[3], v01[3], v02[3], v12[3];

  int i0 = atom->map(shake_atom[m][0]);
  int i1 = atom->map(shake_atom[m][1]);
  int i2 = atom->map(shake_atom[m][2]);
  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];
  double bond12 = angle_distance[shake_type[m][2]];

  MathExtra::sub3(x[i1],x[i0],r01);
  MathExtra::sub3(x[i2],x[i0],r02);
  MathExtra::sub3(x[i2],x[i1],r12);

  domain->minimum_image(FLERR, r01);
  domain->minimum_image(FLERR, r02);
  domain->minimum_image(FLERR, r12);

  MathExtra::sub3(v[i1],v[i0],v01);
  MathExtra::sub3(v[i2],v[i0],v02);
  MathExtra::sub3(v[i2],v[i1],v12);

  double db1 = fabs(sqrt(MathExtra::dot3(r01,r01)) - bond1);
  double db2 = fabs(sqrt(MathExtra::dot3(r02,r02))-bond2);
  double db12 = fabs(sqrt(MathExtra::dot3(r12,r12))-bond12);

  stat = !checkr || (db1 <= tol && db2 <= tol && db12 <= tol);

  if (derr_max < db1/bond1)    derr_max = db1/bond1;
  if (derr_max < db2/bond2)    derr_max = db2/bond2;
  if (derr_max < db12/bond12)  derr_max = db12/bond12;

#if RATTLE_RAISE_ERROR
  if (!stat) error->one(FLERR,"Coordinate constraints are not satisfied up to desired tolerance ");
#endif

  double dv1 = fabs(MathExtra::dot3(r01,v01));
  double dv2 = fabs(MathExtra::dot3(r02,v02));
  double dv12 = fabs(MathExtra::dot3(r12,v12));

  if (verr_max < dv1)    verr_max = dv1;
  if (verr_max < dv2)    verr_max = dv2;
  if (verr_max < dv12)   verr_max = dv12;

  stat = !checkv || (dv1 <= tol && dv2 <= tol && dv12<= tol);

#if RATTLE_RAISE_ERROR
  if (!stat) error->one(FLERR,"Velocity constraints are not satisfied up to desired tolerance!");
#endif
  return stat;
}
