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

#ifndef LMP_BOSONIC_EXCHANGE_NEW_H
#define LMP_BOSONIC_EXCHANGE_NEW_H

#include "pointers.h"
#include <vector>

namespace LAMMPS_NS {

class BosonicExchangeNew : public Pointers {
 public:
  enum RejectMethod { REJECT_NONE, REJECT_DISTANCE, REJECT_PROB };

  BosonicExchangeNew(class LAMMPS *, int, int, int, bool, bool, class RanMars *);
  ~BosonicExchangeNew();

  void prepare_with_coordinates(const double *, const double *, const double *, double, double);
  void spring_force(double **) const;
  double get_bead_spring_energy() const;
  double prim_estimator();

  // Configuration for rejection method
  void set_rejection_params(RejectMethod method, double cutoff, double prob_cutoff, double prob_recover);

 private:
  int nbosons;
  int np;
  int bead_num;
  bool apply_minimum_image;
  bool physical_beta_convention;
  double beta;
  double spring_constant;

  class RanMars *random; // Pointer to RNG from Fix

  // Rejection parameters
  RejectMethod reject_method;
  double cutoff_sq; // Store squared cutoff for distance
  double prob_cutoff;
  double prob_recover;

  // Rejection state
  std::vector<int> rejections;
  std::vector<double> prob_backwards_storage;

  const double *x;
  const double *x_prev;
  const double *x_next;

  double *E_kn;
  double *V;
  double *V_backwards;
  double *connection_probabilities;
  double *temp_nbosons_array;

  void diff_two_beads(const double *, int, const double *, int, double[3]) const;
  double distance_squared_two_beads(const double *, int, const double *, int) const;

  void update_rejections();
  void evaluate_cycle_energies();
  void Evaluate_VBn();
  void Evaluate_V_backwards();
  void evaluate_connection_probabilities();

  double get_Enk(int, int) const;
  void set_Enk(int, int, double);

  double get_potential() const;
  double get_interior_bead_spring_energy() const;

  void spring_force_last_bead(double **) const;
  void spring_force_first_bead(double **) const;
  void spring_force_interior_bead(double **) const;
};

}    // namespace LAMMPS_NS

#endif
