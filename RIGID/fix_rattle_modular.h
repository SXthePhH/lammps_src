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
   fix rattle/modular — GROMACS-style constraint solver modules.

   Provides two standalone iterative constraint modules:
     cshake_module()  — linearized SHAKE, corrects positions directly
     crattle_module() — RATTLE, corrects velocities directly

   Neither module adds constraint forces to f[]. Both follow the
   GROMACS 2024.1 shake.cpp logic (cshake at line 259, crattle at line 349).

   These are designed to be called from other fixes' initial_integrate
   or final_integrate methods, NOT from post_force.

   Reference:
     Ryckaert et al. 1977, J. Comput. Phys. 23, 327
     Andersen 1983, J. Comput. Phys. 52, 24
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(rattle/modular,FixRattleModular);
// clang-format on
#else

#ifndef LMP_FIX_RATTLE_MODULAR_H
#define LMP_FIX_RATTLE_MODULAR_H

#include "fix_shake.h"

namespace LAMMPS_NS {

class FixRattleModular : public FixShake {
 public:
  FixRattleModular(class LAMMPS *, int, char **);
  ~FixRattleModular() override;

  int setmask() override;
  void init() override;
  void setup(int) override;

  void correct_coordinates(int vflag) override;
  void correct_velocities() override;
  void shake_end_of_step(int vflag) override;

  double memory_usage() override;
  void grow_arrays(int) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  // ---- GROMACS-style constraint solver modules ----

  /*! \brief SHAKE for positions — linearized iterative direct position correction.
   *
   *  Reference positions xref provide r₀ (bond vector before correction).
   *  Positions x are corrected in-place.
   *
   *  Per constraint (i,j), per iteration:
   *    diff = d² - |r'|²
   *    Δλ = ω · diff · (μ/2) / (r₀ · r')
   *    x[i] += Δλ · r₀ · m_i⁻¹
   *    x[j] -= Δλ · r₀ · m_j⁻¹
   *
   *  Returns the number of iterations taken. Also stores accumulated
   *  Lagrange multipliers in cb_lambda[], which can be used for
   *  velocity correction:  v[i] += cb_lambda[k] · m_i⁻¹ / Δt · r₀.
   */
  int cshake_module(double **x, double **xref);

  /*! \brief RATTLE for velocities — iterative direct velocity correction.
   *
   *  Reference positions x provide r₀ (constrained bond vector).
   *  Velocities v are corrected in-place.
   *
   *  Per constraint (i,j), per iteration:
   *    v_ij = v[i] - v[j]
   *    Δλ = -ω · (μ / d²) · (v_ij · r₀)
   *    v[i] += Δλ · r₀ · m_i⁻¹
   *    v[j] -= Δλ · r₀ · m_j⁻¹
   *
   *  Returns the number of iterations taken.
   */
  int crattle_module(double **v, double **x);

  // ---- Access to accumulated Lagrange multipliers (for virial / velocity fixup) ----

  int get_num_bonds() const { return nb; }
  const double *get_lambda_array() const { return cb_lambda; }
  const double *get_r0x_array() const { return cb_r0x; }
  const double *get_r0y_array() const { return cb_r0y; }
  const double *get_r0z_array() const { return cb_r0z; }
  const int *get_bond_i_array() const { return cb_i; }
  const int *get_bond_j_array() const { return cb_j; }

  // ---- GROMACS parameters ----
  double omega;            // over-relaxation factor (default 1.0, adaptive via SOR)
  double shake_tol;        // constraint tolerance
  int max_iter;            // max iterations per module call

 private:
  /*! \brief Build per-constraint bond arrays from the current SHAKE cluster
   *  list. Flattens clusters into individual bonds:
   *    flag=2 → 1 bond, flag=3 → 2 bonds, flag=4 → 3 bonds,
   *    flag=1 (angle) → 3 bonds (2 bonds + 1 angle distance).
   */
  void build_constraint_bonds(double **refpos);

  // ---- Per-constraint bond arrays (allocated in build_constraint_bonds) ----

  int nb;                  // total number of constraint bonds
  int *cb_i;               // local atom index i for each bond
  int *cb_j;               // local atom index j for each bond
  double *cb_dsq;          // squared equilibrium distance d²
  double *cb_m2;           // half reduced mass μ/2 = 1/(2*(inv_i+inv_j))
  double *cb_inv_i;        // inverse mass of atom i
  double *cb_inv_j;        // inverse mass of atom j
  double *cb_tolsq;        // convergence tolerance factor = 0.5/(d²*tol)
  double *cb_lambda;       // accumulated Lagrange multiplier (output)
  double *cb_r0x;          // reference bond vector r₀[x]
  double *cb_r0y;          // reference bond vector r₀[y]
  double *cb_r0z;          // reference bond vector r₀[z]

  // ---- SOR adaptation state (GROMACS shake.cpp:783-789) ----
  bool omega_adaptive;     // whether to use SOR adaptation
  double omega_delta;      // SOR step size
  int omega_gamma;         // total work estimate
};

}    // namespace LAMMPS_NS

#endif
#endif
