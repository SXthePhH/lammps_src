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
   fix rattle/modular — GROMACS-style iterative constraint solver modules.

   cshake_module()  (cshake in GROMACS shake.cpp:259)
     Linearized SHAKE (Ryckaert et al. 1977, eq. 5.6).
     Corrects positions x in-place using reference positions xref.

   crattle_module() (crattle in GROMACS shake.cpp:349)
     RATTLE velocity constraint (Andersen 1983).
     Corrects velocities v in-place using constrained positions x.

   Neither module touches f[] (forces). Both accumulate Lagrange
   multipliers in cb_lambda[] for optional virial / velocity fixup.
------------------------------------------------------------------------- */

#include "fix_rattle_modular.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathExtra;

static constexpr double MYTOL  = 1.0e-10;
static constexpr int    MAXNIT = 1000;

/* ---------------------------------------------------------------------- */

FixRattleModular::FixRattleModular(LAMMPS *lmp, int narg, char **arg) :
  FixShake(lmp, narg, arg)
{
  rattle = 1;

  nb        = 0;
  cb_i      = nullptr;
  cb_j      = nullptr;
  cb_dsq    = nullptr;
  cb_m2     = nullptr;
  cb_inv_i  = nullptr;
  cb_inv_j  = nullptr;
  cb_tolsq  = nullptr;
  cb_lambda = nullptr;
  cb_r0x    = nullptr;
  cb_r0y    = nullptr;
  cb_r0z    = nullptr;

  omega          = 1.0;
  shake_tol      = 1.0e-4;
  max_iter       = MAXNIT;
  omega_adaptive = true;
  omega_delta    = 0.1;
  omega_gamma    = 0;
}

/* ---------------------------------------------------------------------- */

FixRattleModular::~FixRattleModular()
{
  memory->destroy(cb_i);
  memory->destroy(cb_j);
  memory->destroy(cb_dsq);
  memory->destroy(cb_m2);
  memory->destroy(cb_inv_i);
  memory->destroy(cb_inv_j);
  memory->destroy(cb_tolsq);
  memory->destroy(cb_lambda);
  memory->destroy(cb_r0x);
  memory->destroy(cb_r0y);
  memory->destroy(cb_r0z);
}

/* ---------------------------------------------------------------------- */

int FixRattleModular::setmask()
{
  int mask = 0;
  mask |= PRE_NEIGHBOR;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRattleModular::init()
{
  FixShake::init();
}

/* ---------------------------------------------------------------------- */

void FixRattleModular::setup(int vflag)
{
  pre_neighbor();
  correct_coordinates(vflag);
  correct_velocities();
  shake_end_of_step(vflag);
}

/* ---------------------------------------------------------------------- */

double FixRattleModular::memory_usage()
{
  double bytes = FixShake::memory_usage();
  if (nb > 0)
    bytes += (double)nb * (6 * sizeof(int) + 10 * sizeof(double));
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixRattleModular::grow_arrays(int nmax)
{
  FixShake::grow_arrays(nmax);
}

/* ---------------------------------------------------------------------- */

int FixRattleModular::pack_forward_comm(int n, int *list, double *buf,
                                        int pbc_flag, int *pbc)
{
  return FixShake::pack_forward_comm(n, list, buf, pbc_flag, pbc);
}

/* ---------------------------------------------------------------------- */

void FixRattleModular::unpack_forward_comm(int n, int first, double *buf)
{
  FixShake::unpack_forward_comm(n, first, buf);
}

/* ======================================================================
   build_constraint_bonds — flattens SHAKE clusters into flat bond list
   ====================================================================== */

void FixRattleModular::build_constraint_bonds(double **refpos)
{
  int total = 0;
  for (int ii = 0; ii < nlist; ii++) {
    int m = list[ii];
    if (shake_flag[m] == 2)
      total += 1;
    else if (shake_flag[m] == 3)
      total += 2;
    else if (shake_flag[m] == 4)
      total += 3;
    else if (shake_flag[m] == 1)
      total += 3;
  }

  if (total > nb) {
    memory->destroy(cb_i);
    memory->destroy(cb_j);
    memory->destroy(cb_dsq);
    memory->destroy(cb_m2);
    memory->destroy(cb_inv_i);
    memory->destroy(cb_inv_j);
    memory->destroy(cb_tolsq);
    memory->destroy(cb_lambda);
    memory->destroy(cb_r0x);
    memory->destroy(cb_r0y);
    memory->destroy(cb_r0z);

    nb = total;
    if (nb > 0) {
      memory->create(cb_i,      nb, "rattle_modular:cb_i");
      memory->create(cb_j,      nb, "rattle_modular:cb_j");
      memory->create(cb_dsq,    nb, "rattle_modular:cb_dsq");
      memory->create(cb_m2,     nb, "rattle_modular:cb_m2");
      memory->create(cb_inv_i,  nb, "rattle_modular:cb_inv_i");
      memory->create(cb_inv_j,  nb, "rattle_modular:cb_inv_j");
      memory->create(cb_tolsq,  nb, "rattle_modular:cb_tolsq");
      memory->create(cb_lambda, nb, "rattle_modular:cb_lambda");
      memory->create(cb_r0x,    nb, "rattle_modular:cb_r0x");
      memory->create(cb_r0y,    nb, "rattle_modular:cb_r0y");
      memory->create(cb_r0z,    nb, "rattle_modular:cb_r0z");
    }
  }

  int kk = 0;
  for (int ii = 0; ii < nlist; ii++) {
    int m    = list[ii];
    int flag = shake_flag[m];

    int i0 = closest_list[ii][0];
    int i1 = (flag >= 2)              ? closest_list[ii][1] : -1;
    int i2 = (flag == 3 || flag == 1) ? closest_list[ii][2] : -1;
    int i3 = (flag == 4)              ? closest_list[ii][3] : -1;

    double inv[4] = { 0.0, 0.0, 0.0, 0.0 };
    if (rmass) {
      inv[0] = 1.0 / rmass[i0];
      if (i1 >= 0) inv[1] = 1.0 / rmass[i1];
      if (i2 >= 0) inv[2] = 1.0 / rmass[i2];
      if (i3 >= 0) inv[3] = 1.0 / rmass[i3];
    } else {
      inv[0] = 1.0 / mass[type[i0]];
      if (i1 >= 0) inv[1] = 1.0 / mass[type[i1]];
      if (i2 >= 0) inv[2] = 1.0 / mass[type[i2]];
      if (i3 >= 0) inv[3] = 1.0 / mass[type[i3]];
    }

    if (flag == 2) {
      // 2-atom cluster: 1 bond (0,1)
      cb_i[kk]   = i0;
      cb_j[kk]   = i1;
      cb_dsq[kk] = bond_distance[shake_type[m][0]] * bond_distance[shake_type[m][0]];
      cb_inv_i[kk] = inv[0];
      cb_inv_j[kk] = inv[1];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[1]));
      kk++;
    } else if (flag == 3) {
      // 3-atom cluster: 2 bonds (0,1), (0,2)
      cb_i[kk]   = i0; cb_j[kk] = i1;
      cb_dsq[kk] = bond_distance[shake_type[m][0]] * bond_distance[shake_type[m][0]];
      cb_inv_i[kk] = inv[0]; cb_inv_j[kk] = inv[1];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[1]));
      kk++;
      cb_i[kk]   = i0; cb_j[kk] = i2;
      cb_dsq[kk] = bond_distance[shake_type[m][1]] * bond_distance[shake_type[m][1]];
      cb_inv_i[kk] = inv[0]; cb_inv_j[kk] = inv[2];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[2]));
      kk++;
    } else if (flag == 4) {
      // 4-atom cluster: 3 bonds (0,1), (0,2), (0,3)
      cb_i[kk]   = i0; cb_j[kk] = i1;
      cb_dsq[kk] = bond_distance[shake_type[m][0]] * bond_distance[shake_type[m][0]];
      cb_inv_i[kk] = inv[0]; cb_inv_j[kk] = inv[1];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[1]));
      kk++;
      cb_i[kk]   = i0; cb_j[kk] = i2;
      cb_dsq[kk] = bond_distance[shake_type[m][1]] * bond_distance[shake_type[m][1]];
      cb_inv_i[kk] = inv[0]; cb_inv_j[kk] = inv[2];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[2]));
      kk++;
      cb_i[kk]   = i0; cb_j[kk] = i3;
      cb_dsq[kk] = bond_distance[shake_type[m][2]] * bond_distance[shake_type[m][2]];
      cb_inv_i[kk] = inv[0]; cb_inv_j[kk] = inv[3];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[3]));
      kk++;
    } else if (flag == 1) {
      // angle cluster (3 atoms): 2 bonds + 1 angle distance
      // bond (0,1)
      cb_i[kk]   = i0; cb_j[kk] = i1;
      cb_dsq[kk] = bond_distance[shake_type[m][0]] * bond_distance[shake_type[m][0]];
      cb_inv_i[kk] = inv[0]; cb_inv_j[kk] = inv[1];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[1]));
      kk++;
      // bond (0,2)
      cb_i[kk]   = i0; cb_j[kk] = i2;
      cb_dsq[kk] = bond_distance[shake_type[m][1]] * bond_distance[shake_type[m][1]];
      cb_inv_i[kk] = inv[0]; cb_inv_j[kk] = inv[2];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[0] + inv[2]));
      kk++;
      // angle distance (1,2)
      cb_i[kk]   = i1; cb_j[kk] = i2;
      cb_dsq[kk] = angle_distance[shake_type[m][2]] * angle_distance[shake_type[m][2]];
      cb_inv_i[kk] = inv[1]; cb_inv_j[kk] = inv[2];
      cb_m2[kk]  = 1.0 / (2.0 * (inv[1] + inv[2]));
      kk++;
    }
  }

  // compute r₀ (reference bond vectors) and tolerance factors
  for (int k = 0; k < nb; k++) {
    cb_r0x[k] = refpos[cb_i[k]][0] - refpos[cb_j[k]][0];
    cb_r0y[k] = refpos[cb_i[k]][1] - refpos[cb_j[k]][1];
    cb_r0z[k] = refpos[cb_i[k]][2] - refpos[cb_j[k]][2];
    domain->minimum_image(FLERR, cb_r0x[k], cb_r0y[k], cb_r0z[k]);
    cb_tolsq[k] = 0.5 / (cb_dsq[k] * shake_tol);
  }
}

/* ======================================================================
   GROMACS-STYLE CONSTRAINT SOLVER MODULES
   ====================================================================== */

/* ----------------------------------------------------------------------
   cshake_module — linearized SHAKE, corrects positions directly.

   GROMACS equivalent: cshake() at shake.cpp:259-346

   Parameters:
     x    — positions to correct (modified in-place)
     xref — reference (unconstrained) positions, used for r₀

   Returns number of iterations taken. Zero or negative means no work.
   Accumulates Lagrange multipliers in cb_lambda[].

   Per-constraint update (each iteration, sequential over bonds):
     diff = d² - |r'|²
     Δλ  = ω · diff · (μ/2) / (r₀ · r')
     x[i] += Δλ · r₀ · m_i⁻¹
     x[j] -= Δλ · r₀ · m_j⁻¹

   After return, caller can correct velocities (GROMACS vec_shakef:567-581):
     invdt = 1.0 / Δt;
     v[i] += cb_lambda[k] · m_i⁻¹ · invdt · r₀[k]
     v[j] -= cb_lambda[k] · m_j⁻¹ · invdt · r₀[k]
------------------------------------------------------------------------- */

int FixRattleModular::cshake_module(double **x, double **xref)
{
  if (nlist == 0) return 0;

  build_constraint_bonds(xref);

  for (int k = 0; k < nb; k++)
    cb_lambda[k] = 0.0;

  int nconv  = 1;
  int error  = 0;
  int nit    = 0;
  int maxnit = (omega_adaptive) ? MAXNIT : max_iter;

  while (nit < maxnit && nconv != 0 && error == 0) {
    nconv = 0;
    for (int k = 0; k < nb && error == 0; k++) {

      double dx = x[cb_i[k]][0] - x[cb_j[k]][0];
      double dy = x[cb_i[k]][1] - x[cb_j[k]][1];
      double dz = x[cb_i[k]][2] - x[cb_j[k]][2];
      domain->minimum_image(FLERR, dx, dy, dz);

      double r_prime_sq = dx * dx + dy * dy + dz * dz;
      double diff       = cb_dsq[k] - r_prime_sq;

      double iconvf = fabs(diff) * cb_tolsq[k];

      if (iconvf > 1.0) {
        nconv = static_cast<int>(iconvf);

        double r_dot_r_prime =
            cb_r0x[k] * dx + cb_r0y[k] * dy + cb_r0z[k] * dz;

        if (r_dot_r_prime < cb_dsq[k] * MYTOL) {
          error = k + 1;
        } else {
          double lambda_inc = omega * diff * cb_m2[k] / r_dot_r_prime;
          cb_lambda[k] += lambda_inc;

          x[cb_i[k]][0] += lambda_inc * cb_r0x[k] * cb_inv_i[k];
          x[cb_i[k]][1] += lambda_inc * cb_r0y[k] * cb_inv_i[k];
          x[cb_i[k]][2] += lambda_inc * cb_r0z[k] * cb_inv_i[k];

          x[cb_j[k]][0] -= lambda_inc * cb_r0x[k] * cb_inv_j[k];
          x[cb_j[k]][1] -= lambda_inc * cb_r0y[k] * cb_inv_j[k];
          x[cb_j[k]][2] -= lambda_inc * cb_r0z[k] * cb_inv_j[k];
        }
      }
    }
    nit++;
  }

  if (omega_adaptive && nit > 0) {
    int total_ops = nit * nb;
    if (total_ops > omega_gamma)
      omega_delta *= -0.5;
    omega += omega_delta;
    omega_gamma = total_ops;
  }

  return (nit < maxnit && error == 0) ? nit : -nit;
}

/* ----------------------------------------------------------------------
   crattle_module — RATTLE for velocities, corrects velocities directly.

   GROMACS equivalent: crattle() at shake.cpp:349-422

   Parameters:
     v — velocities to correct (modified in-place)
     x — constrained positions, used for reference bond vectors r₀

   Returns number of iterations taken.

   Per-constraint update (each iteration, sequential over bonds):
     v_ij = v[i] - v[j]
     Δλ  = -ω · (μ / d²) · (v_ij · r₀)
     v[i] += Δλ · r₀ · m_i⁻¹
     v[j] -= Δλ · r₀ · m_j⁻¹
------------------------------------------------------------------------- */

int FixRattleModular::crattle_module(double **v, double **x)
{
  if (nlist == 0) return 0;

  build_constraint_bonds(x);

  for (int k = 0; k < nb; k++)
    cb_lambda[k] = 0.0;

  double invdt = 1.0 / update->dt;

  int nconv  = 1;
  int error  = 0;
  int nit    = 0;
  int maxnit = (omega_adaptive) ? MAXNIT : max_iter;

  while (nit < maxnit && nconv != 0 && error == 0) {
    nconv = 0;
    for (int k = 0; k < nb && error == 0; k++) {

      double dvx = v[cb_i[k]][0] - v[cb_j[k]][0];
      double dvy = v[cb_i[k]][1] - v[cb_j[k]][1];
      double dvz = v[cb_i[k]][2] - v[cb_j[k]][2];

      double vpijd  = dvx * cb_r0x[k] + dvy * cb_r0y[k] + dvz * cb_r0z[k];
      double iconvf = fabs(vpijd) * cb_tolsq[k] * invdt;

      if (iconvf > 1.0) {
        nconv = static_cast<int>(iconvf);

        double acor = -omega * (2.0 * cb_m2[k] / cb_dsq[k]) * vpijd;
        cb_lambda[k] += acor;

        v[cb_i[k]][0] += acor * cb_r0x[k] * cb_inv_i[k];
        v[cb_i[k]][1] += acor * cb_r0y[k] * cb_inv_i[k];
        v[cb_i[k]][2] += acor * cb_r0z[k] * cb_inv_i[k];

        v[cb_j[k]][0] -= acor * cb_r0x[k] * cb_inv_j[k];
        v[cb_j[k]][1] -= acor * cb_r0y[k] * cb_inv_j[k];
        v[cb_j[k]][2] -= acor * cb_r0z[k] * cb_inv_j[k];
      }
    }
    nit++;
  }

  if (omega_adaptive && nit > 0) {
    int total_ops = nit * nb;
    if (total_ops > omega_gamma)
      omega_delta *= -0.5;
    omega += omega_delta;
    omega_gamma = total_ops;
  }

  return (nit < maxnit && error == 0) ? nit : -nit;
}

/* ======================================================================
   LAMMPS integration hooks (called during setup)
   ====================================================================== */

/* ----------------------------------------------------------------------
   correct_coordinates — GROMACS-style position correction for setup.
   Uses unconstrained_update to predict positions into xshake,
   then calls cshake_module to correct x in-place.
------------------------------------------------------------------------- */

void FixRattleModular::correct_coordinates(int /*vflag*/)
{
  if (nlist == 0) return;

  unconstrained_update();

  if (comm->nprocs > 1)
    comm->forward_comm(this);

  cshake_module(atom->x, xshake);
}

/* ----------------------------------------------------------------------
   correct_velocities — GROMACS-style velocity correction for setup.
   Uses crattle_module directly on current velocities.
------------------------------------------------------------------------- */

void FixRattleModular::correct_velocities()
{
  if (nlist == 0) return;

  if (comm->nprocs > 1)
    comm->forward_comm(this);

  crattle_module(atom->v, atom->x);
}

/* ---------------------------------------------------------------------- */

void FixRattleModular::shake_end_of_step(int /*vflag*/)
{
  // no-op — constraint modules are called directly by other fixes
}

/* ---------------------------------------------------------------------- */

// void FixRattleModular::reset_dt()
// {
//   FixShake::reset_dt();
// }
