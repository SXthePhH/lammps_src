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

#include "bosonic_exchange_new.h"

#include "domain.h"
#include "error.h"
#include "memory.h"
#include "random_mars.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace LAMMPS_NS;

BosonicExchangeNew::BosonicExchangeNew(LAMMPS *lmp, int nbosons, int np, int bead_num, bool mic,
                                 bool beta_convention, RanMars *rng) :
    Pointers(lmp), nbosons(nbosons), np(np), bead_num(bead_num), apply_minimum_image(mic),
    physical_beta_convention(beta_convention), random(rng), x(nullptr), x_prev(nullptr), x_next(nullptr),
    E_kn(nullptr), V(nullptr), V_backwards(nullptr), connection_probabilities(nullptr),
    temp_nbosons_array(nullptr)
{
  memory->create(temp_nbosons_array, nbosons + 1, "BosonicExchangeNew: temp_nbosons_array");
  memory->create(E_kn, (nbosons * (nbosons + 1) / 2), "BosonicExchangeNew: E_kn");
  memory->create(V, nbosons + 1, "BosonicExchangeNew: V");
  memory->create(V_backwards, nbosons + 1, "BosonicExchangeNew: V_backwards");
  memory->create(connection_probabilities, nbosons * nbosons,
                 "BosonicExchangeNew: connection probabilities");

  reject_method = REJECT_DISTANCE; // Default
  cutoff_sq = 1e8;
  prob_cutoff = 1e-7;
  prob_recover = 0.1;
  prob_backwards_storage.resize(nbosons, 1.0); // Initialize with 1.0 (fully connected)
  
  rejections.push_back(0);
  rejections.push_back(nbosons);
}

BosonicExchangeNew::~BosonicExchangeNew()
{
  memory->destroy(connection_probabilities);
  memory->destroy(V_backwards);
  memory->destroy(V);
  memory->destroy(E_kn);
  memory->destroy(temp_nbosons_array);
}

void BosonicExchangeNew::set_rejection_params(RejectMethod method, double cutoff, double p_cutoff, double p_recover) {
    reject_method = method;
    cutoff_sq = cutoff * cutoff;
    prob_cutoff = p_cutoff;
    prob_recover = p_recover;
}

void BosonicExchangeNew::update_rejections() {
    rejections.clear();
    rejections.push_back(0);

    if (reject_method == REJECT_DISTANCE) {
        // Based on distance between x_prev[i] (bead P-1) and x[i+1] (bead 0)
        // This corresponds to the link (i, P-1) -> (i+1, 0)
        for (int i = 0; i < nbosons - 1; ++i) {
            double diff[3];
            // x_prev is effectively bead P-1 (from fix)
            // x is bead 0
            // We want dist between bead P-1 of particle i and bead 0 of particle i+1
            diff_two_beads(x_prev, i, x, i + 1, diff);
            double r2 = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
            if (r2 > cutoff_sq) {
                // Reject connection between i and i+1
                // This means index i+1 is a start of a new segment
                rejections.push_back(i + 1);
            }
        }
    } else if (reject_method == REJECT_PROB) {
        for (int i = 1; i < nbosons; ++i) {
            // prob_backwards_storage[i] stores the probability of link (i-1) -> i
            if (prob_backwards_storage[i] < prob_cutoff) {
                rejections.push_back(i);
            }
        }
    }

    rejections.push_back(nbosons);

    // Random recover
    if (rejections.size() > 2 && prob_recover > 0.0 && random) {
        std::vector<int> new_rejections;
        new_rejections.push_back(0);
        
        for (size_t k = 1; k < rejections.size() - 1; ++k) {
            double rnd = random->uniform();
            if (rnd > prob_recover) {
                // Keep the rejection
                new_rejections.push_back(rejections[k]);
            } 
            // else: Recover (do not add to new_rejections)
        }
        
        new_rejections.push_back(nbosons);
        rejections = new_rejections;
    }
}

void BosonicExchangeNew::prepare_with_coordinates(const double *x, const double *x_prev,
                                               const double *x_next, double beta,
                                               double spring_constant)
{
  this->x = x;
  this->x_prev = x_prev;
  this->x_next = x_next;
  this->beta = beta;
  this->spring_constant = spring_constant;

  if (bead_num == 0 || bead_num == np - 1) {
    // exterior beads
    // Update rejections before evaluating energies
    update_rejections();

    evaluate_cycle_energies();
    Evaluate_VBn();
    Evaluate_V_backwards();
    evaluate_connection_probabilities();
  }
}

void BosonicExchangeNew::diff_two_beads(const double *x1, int l1, const double *x2, int l2,
                                     double diff[3]) const
{
  l1 = l1 % nbosons;
  l2 = l2 % nbosons;
  double delx2 = x2[3 * l2 + 0] - x1[3 * l1 + 0];
  double dely2 = x2[3 * l2 + 1] - x1[3 * l1 + 1];
  double delz2 = x2[3 * l2 + 2] - x1[3 * l1 + 2];
  if (apply_minimum_image) { domain->minimum_image(FLERR, delx2, dely2, delz2); }

  diff[0] = delx2;
  diff[1] = dely2;
  diff[2] = delz2;
}


double BosonicExchangeNew::distance_squared_two_beads(const double *x1, int l1, const double *x2,
                                                   int l2) const
{
  double diff[3];
  diff_two_beads(x1, l1, x2, l2, diff);
  return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
}


void BosonicExchangeNew::evaluate_cycle_energies()
{
  const double *x_first_bead;
  const double *x_last_bead;

  if (bead_num == 0) {
    x_first_bead = x;
    x_last_bead = x_prev;
  } else {
    x_first_bead = x_next;
    x_last_bead = x;
  }

  for (int i = 0; i < nbosons; i++) {
    temp_nbosons_array[i] = distance_squared_two_beads(x_first_bead, i, x_last_bead, i);
  }

  // Iterate through segments defined by rejections
  for (size_t seg = 0; seg < rejections.size() - 1; ++seg) {
      int start = rejections[seg];
      int end = rejections[seg+1];

      // Original loop: for v = 1 to N.
      // Now v runs from start+1 to end? No, rejections contain indices of particles.
      // Particle indices are 0-based: 0 to nbosons-1.
      // v in original code is 1-based index (0 to nbosons-1 loop for v=i+1?? No).
      // Original: for (int v = 0; v < nbosons; v++) -> set_Enk(v+1, ...)
      // So v+1 goes from 1 to nbosons.
      
      for (int v_idx = start; v_idx < end; ++v_idx) {
          int v_1based = v_idx + 1;
          
          set_Enk(v_1based, 1, 0.5 * spring_constant * (temp_nbosons_array[v_idx]));

          // Inner loop u goes down from v_idx - 1.
          // Must stay within [start, end) segment.
          // u cannot go below start.
          for (int u_idx = v_idx - 1; u_idx >= start; u_idx--) {
              double val = get_Enk(v_1based, v_idx - u_idx) +
                  0.5 * spring_constant *
                      (
                          +distance_squared_two_beads(x_last_bead, u_idx, x_first_bead, u_idx + 1)
                          - distance_squared_two_beads(x_first_bead, u_idx + 1, x_last_bead, v_idx)
                          + distance_squared_two_beads(x_first_bead, u_idx, x_last_bead, v_idx));

              set_Enk(v_1based, v_idx - u_idx + 1, val);
          }
      }
  }
}

double BosonicExchangeNew::get_Enk(int m, int k) const
{
  int end_of_m = m * (m + 1) / 2;
  return E_kn[end_of_m - k];
}

void BosonicExchangeNew::set_Enk(int m, int k, double val)
{
  int end_of_m = m * (m + 1) / 2;
  E_kn[end_of_m - k] = val;
}

void BosonicExchangeNew::Evaluate_VBn()
{
  V[0] = 0.0;
  
  // Iterate segments
  for (size_t seg = 0; seg < rejections.size() - 1; ++seg) {
      int start = rejections[seg];
      int end = rejections[seg+1];
      
      // Original: for (int m = 1; m < nbosons + 1; m++)
      // Here we compute V for m in range [start+1, end] (relative to global 0? V is global size N+1)
      // V[m] depends on V[m-k].
      // If m is in [start+1, end], V[m] should only connect to beads >= start.
      // So k can go up to m - start.
      
      for (int m = start + 1; m <= end; m++) {
        double Elongest = std::numeric_limits<double>::max();
        
        // k runs from m down to 1. But restricted by start.
        // The farthest back we can go is start.
        // So m - k >= start  =>  k <= m - start.
        // Max k is m - start.
        int k_max = m - start;

        for (int k = m - start; k > 0; k--) {
          double val = get_Enk(m, k) + V[m - k];
            Elongest = std::min(Elongest, val);
            temp_nbosons_array[k - 1] = val;
        }

        double sig_denom = 0.0;
        for (int k = m - start; k > 0; k--) {
            sig_denom += exp(-beta * (temp_nbosons_array[k - 1] - Elongest));
        }
        
        // V[m] stores log partition function up to m.
        // The "m" in division inside log(sig_denom / m) depends on recursion relative to block start?
        // Original code: V[m] = ... - log(sig_denom / m).
        // The factor 1/m comes from cycle weight. Global index m?
        // In the original derivation, V[N] is approximations to Z.
        // The term 1/N is for the cycle permutations. 
        
        // In segmented approach, we are effectively calculating small Z for each segment and multiplying them?
        // Actually, V usually accumulates. 
        // If we have full separation (block diagonal), Z_total = Z_1 * Z_2 ...
        // log Z_total = log Z_1 + log Z_2 ...
        // Here we run V sequentially. V[end_1] will be log Z_1.
        // Then for next segment [start2, end2], V[start2] is V[end_1].
        // We want V[k] = V[start2] + delta_V.
        // So the basic recurrence stands: V[m] = ... + V[m-k].
        // BUT the factor 1/m should be 1/length_of_cycle?
        // get_Enk(m, k) is energy of cycle of length k.
        
        // In recursion: Z_N = sum_{k=1}^N Z_{N-k} * exp(-beta E(N,k)) / N ???
        // Hirshberg 2023 paper eq.
        // Yes, the factor is 1/N in the recurrence for standard Bosonic.
        // But here we are in a segment.
        // If we treat the segment as an isolated system of N' bosons, the factor should be 1/m' where m' is index within segment?
        // Or can we trust the global index?
        // The factor 1/N comes from N indistinguishable particles.
        // The recurrence is: Q_N = 1/N * sum_{k=1}^N exp(-beta E_{k}) Q_{N-k}.
        // If we split into subsystems N1 and N2. Q_N = Q_N1 * Q_N2.
        // Q_m in the loop calculation for segment 2 (relative to start of seg 2) should use local index?
        // Let's assume we use local index relative to start of segment.
        // m_local = m - start.
        // But what about V[m-k]? If k=m_local, V[m-k] = V[start].
        // Ideally V[start] should be added as a constant offset log(Z_prev_blocks).
        // So V[m] = log(Z_{current_block_up_to_m_local}) + V[start].
        
        // Let's modify the 1/m factor to 1/(m - start).
        // WARNING: This is a physical assumption. If we reject exchanges, we are saying particles in standard order [0..N] are indistinguishable only within blocks.
        // Effectively we have species A (size N1), species B (size N2)...
        // So they obey Boltzmann statistics for N1 bosons, N2 bosons.
        // So yes, 1/(m - start) is correct for the sub-block.
        
        int m_local = m - start;
        
        V[m] = Elongest - (1.0 / beta) * log(sig_denom / (double) m_local);

        if (!std::isfinite(V[m])) {
          error->universe_one(
              FLERR,
              fmt::format("Invalid sig_denom {} with Elongest {} in bosonic exchange potential",
                          sig_denom, Elongest));
        }
      }
  }
}

void BosonicExchangeNew::Evaluate_V_backwards()
{
  V_backwards[nbosons] = 0.0;
  // Initialize probability storage for Probed Rejection
  std::fill(prob_backwards_storage.begin(), prob_backwards_storage.end(), 0.0);

  // Iterate segments backwards
  for (int seg = rejections.size() - 2; seg >= 0; --seg) {
      int start = rejections[seg];
      int end = rejections[seg+1];
      
      // V_backwards[end] is 0 relative to end of block?
      // Wait, V_backwards[l] accumulates from right to left?
      // In original code, V_back[l] uses V_back[p+1].
      // For block [start, end), we have boundary condition V_back[end].
      // If independent blocks, V_back[end] should essentially restart (be 0 relative to block).
      // But we need to carry over the total potential if we want global continuity (though strictly only diffs matter for prob?).
      // For probabilities: P ~ exp( -beta (V + V_back - V_total) ).
      // If V accumulates forward, V[end] includes Potentials of block 1..block i.
      // If V_back accumulates backward, V_back[start] includes Potentials of block i..block M.
      // So V[l] + V_back[l] should be roughly V_total.
      
      // Let's set V_backwards[end] = V_backwards[global_end] ... 
      // Actually, if we treat blocks as independent:
      // Z = Z1 * Z2.
      // E.g. block [0, 2], [2, 4].
      // V[2] = log Z1. V[4] = log Z1 + log Z2.
      // V_back[2] should be log Z2?
      // V_back[0] should be log Z1 + log Z2.
      // Let's see original code: V_backwards[0] = V[nbosons].
      // So we can compute V_backwards locally within block, then shift?
      // Or just enforce global accumulation.
      
      // Let's ensure V_backwards[end] is set correctly.
      // For the last segment, V_backwards[nbosons] = 0. (Already set).
      // For other segments (seg < last), V_backwards[end] should be computed?
      // No, V_backwards loop goes l = nbosons - 1 down to 0.
      // We can just execute the loop but restrict p range.
      
      for (int l = end - 1; l >= start; l--) {
            double Elongest = std::numeric_limits<double>::max();
            
            // p runs from l to end - 1. (Cannot cross end).
            // Inner loop:
            int p_max = end - 1;
            
            for (int p = l; p <= p_max; p++) {
                double val = get_Enk(p + 1, p - l + 1) + V_backwards[p + 1];
                Elongest = std::min(Elongest, val);
                temp_nbosons_array[p] = val;
            }

            double sig_denom = 0.0;
            // 1/(p+1) factor correction -> 1/(p - start + 1) similar to forward pass?
            for (int p = l; p <= p_max; p++) {
                int p_local_plus_1 = p - start + 1;
                sig_denom += 1.0 / (p_local_plus_1) * exp(-beta * (temp_nbosons_array[p] - Elongest));
            }

            V_backwards[l] = Elongest - log(sig_denom) / beta;
            
            // Store probability for next step rejection logic
            // The probability of connecting l-1 -> l ?
            // In original code, connection_probabilities stored.
            // prob_backwards usually means probability of having a link/connection versus not.
            // Python: prob_backwards[n] = 1 - exp(...)
            // Let's compute it in 'evaluate_connection_probabilities' and store it there.
            
            if (!std::isfinite(V_backwards[l])) {
              error->universe_one(
                  FLERR,
                  fmt::format(
                      "Invalid sig_denom {} with Elongest {} in bosonic exchange potential backwards",
                      sig_denom, Elongest));
            }
      }
      // V_backwards[start] is now computed.
  }

  V_backwards[0] = V[nbosons];
}

double BosonicExchangeNew::get_potential() const
{
  return V[nbosons];
}

double BosonicExchangeNew::get_interior_bead_spring_energy() const
{
  double spring_energy_for_bead = 0.;
  for (int i = 0; i < nbosons; i++) {
    spring_energy_for_bead += 0.5 * spring_constant * distance_squared_two_beads(x, i, x_prev, i);
  }
  return spring_energy_for_bead;
}

double BosonicExchangeNew::get_bead_spring_energy() const
{
  double spring_energy_for_bead =
      (bead_num == 0 ? get_potential() : get_interior_bead_spring_energy());
  return spring_energy_for_bead;
}

void BosonicExchangeNew::evaluate_connection_probabilities()
{
  // Zero out everything first (important for cut regions)
  std::fill(connection_probabilities, connection_probabilities + nbosons*nbosons, 0.0);

  for (size_t seg = 0; seg < rejections.size() - 1; ++seg) {
      int start = rejections[seg];
      int end = rejections[seg+1];
      
      // l runs from start to end - 2?
      // original: l < nbosons - 1. l is index.
      // direct link prob for l -> l+1.
      // Only valid if l+1 < end.
      
      for (int l = start; l < end - 1; l++) {
        double direct_link_probability =
            1.0 - (exp(-beta * (V[l + 1] + V_backwards[l + 1] - V[nbosons])));
        
        connection_probabilities[nbosons * l + (l + 1)] = direct_link_probability;
        
        // Store for next step rejection
        // Store at index l+1 (prob of link l->l+1 exists at l+1?)
        // Python: prob_backwards[n] = 1 - np.exp(...) for n=1..N
        // Corresponds to link n-1 -> n
        prob_backwards_storage[l+1] = direct_link_probability;
      }
      
      // Cycle probabilities
      for (int u = start; u < end; u++) {
        for (int l = u; l < end; l++) {
          // Factor 1/(l+1) -> 1/(l-start+1)
          int l_local_plus_1 = l - start + 1;
          double close_cycle_probability = 1.0 / (l_local_plus_1) *
              exp(-beta * (V[u] + get_Enk(l + 1, l - u + 1) + V_backwards[l + 1] - V[nbosons]));
          connection_probabilities[nbosons * l + u] = close_cycle_probability;
        }
      }
  }
}


void BosonicExchangeNew::spring_force(double **f) const
{
  if (bead_num == np - 1) {
    spring_force_last_bead(f);
  } else if (bead_num == 0) {
    spring_force_first_bead(f);
  } else {
    spring_force_interior_bead(f);
  }
}


void BosonicExchangeNew::spring_force_last_bead(double **f) const
{
  const double *x_first_bead = x_next;
  const double *x_last_bead = x;

  for (int l = 0; l < nbosons; l++) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    
    // Limits based on rejections?
    // connection_probabilities should be zero outside segments, so full loop is safe but inefficient.
    // Optimization: find segment for l.
    // Since we didn't store segment map, we iterate bounds.
    // But safely, just checking next_l validity is enough if we trust connection_probabilities being 0.
    // However, rejections might strictly forbid connections.
    
    // Let's implement robust finding of end limit for optimization
    int end_limit = nbosons;
    // Find end of segment containing l
    // Could use binary search on rejections.
    // Since l increases, we can track segment. But for simplicity and safety against small N, simple loop.
    // Or just trust connection probabilities are 0.
    
    for (int next_l = 0; next_l <= l + 1 && next_l < nbosons; next_l++) {
      double prob = connection_probabilities[nbosons * l + next_l];
      if (prob == 0.0) continue; 

      double diff_next[3];
      diff_two_beads(x_last_bead, l, x_first_bead, next_l, diff_next);

      sum_x += prob * diff_next[0];
      sum_y += prob * diff_next[1];
      sum_z += prob * diff_next[2];
    }

    double diff_prev[3];
    diff_two_beads(x_last_bead, l, x_prev, l, diff_prev);
    sum_x += diff_prev[0];
    sum_y += diff_prev[1];
    sum_z += diff_prev[2];

    f[l][0] += sum_x * spring_constant;
    f[l][1] += sum_y * spring_constant;
    f[l][2] += sum_z * spring_constant;
  }
}

void BosonicExchangeNew::spring_force_first_bead(double **f) const
{
  const double *x_first_bead = x;
  const double *x_last_bead = x_prev;

  for (int l = 0; l < nbosons; l++) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (int prev_l = std::max(0, l - 1); prev_l < nbosons; prev_l++) {
      double prob = connection_probabilities[nbosons * prev_l + l];
      if (prob == 0.0) continue;

      double diff_prev[3];
      diff_two_beads(x_first_bead, l, x_last_bead, prev_l, diff_prev);

      sum_x += prob * diff_prev[0];
      sum_y += prob * diff_prev[1];
      sum_z += prob * diff_prev[2];
    }

    double diff_next[3];
    diff_two_beads(x_first_bead, l, x_next, l, diff_next);
    sum_x += diff_next[0];
    sum_y += diff_next[1];
    sum_z += diff_next[2];

    f[l][0] += sum_x * spring_constant;
    f[l][1] += sum_y * spring_constant;
    f[l][2] += sum_z * spring_constant;
  }
}

void BosonicExchangeNew::spring_force_interior_bead(double **f) const
{
  for (int l = 0; l < nbosons; l++) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;

    double diff_prev[3];
    diff_two_beads(x, l, x_prev, l, diff_prev);
    sum_x += diff_prev[0];
    sum_y += diff_prev[1];
    sum_z += diff_prev[2];

    double diff_next[3];
    diff_two_beads(x, l, x_next, l, diff_next);
    sum_x += diff_next[0];
    sum_y += diff_next[1];
    sum_z += diff_next[2];

    f[l][0] += sum_x * spring_constant;
    f[l][1] += sum_y * spring_constant;
    f[l][2] += sum_z * spring_constant;
  }
}

double BosonicExchangeNew::prim_estimator()
{
  // Adjusted for segments
  double convention_correction = (physical_beta_convention ? 1 : 1.0 / np);

  if (bead_num != 0) {
    return convention_correction *
        (0.5 * domain->dimension * nbosons / beta - get_interior_bead_spring_energy());
  }

  temp_nbosons_array[0] = 0.0;
  
  // Need to sum over segments? 
  // prim estimator involves sum over permutations?
  // Original is a loop m = 1 to N.
  // temp_nbosons_array stores something like kinetic energy contribution?
  // Hirshberg eq S4-S5.
  // The structure of the loop suggests it can be segmented.
  
  double total_val = 0.0;

  for (size_t seg = 0; seg < rejections.size() - 1; ++seg) {
      int start = rejections[seg];
      int end = rejections[seg+1];
      
      // temp_nbosons_array needs to be handled within segment.
      // Reset temp[start] = 0.0 ? No, logic is recursive or independent?
      // m loop computes term for m.
      // term for m depends on temp[m-k].
      
      // If segments are independent, temp[start] should be initialized to what?
      // Original temp[0] = 0.0.
      // temp_nbosons_array[m] stores numerator for estimator.
      // It seems to be independent?
      // Let's assume start corresponds to local 0.
      
      // We'll map m global to m_local in the array for safety or just use global indices 
      // but ensure temp[start] is treated as boundary.
      
      temp_nbosons_array[start] = 0.0;

      for (int m = start + 1; m <= end; ++m) {
        double sig = 0.0;
        temp_nbosons_array[m] = 0.0;

        double Elongest = std::numeric_limits<double>::max();
        for (int k = m - start; k > 0; k--) { 
            Elongest = std::min(Elongest, get_Enk(m, k) + V[m - k]); 
        }

        for (int k = m - start; k > 0; --k) {
          double E_kn_val = get_Enk(m, k);
          // temp_nbosons_array[m-k] should be valid.
          sig += (temp_nbosons_array[m - k] - E_kn_val) * exp(-beta * (E_kn_val + V[m - k] - Elongest));
        }

        int m_local = m - start;
        double sig_denom_m = m_local * exp(-beta * (V[m] - Elongest));

        if (sig_denom_m != 0.0)
            temp_nbosons_array[m] = sig / sig_denom_m;
        else
            temp_nbosons_array[m] = 0.0;
      }
      
      // Accumulate the value at end of segment
      total_val += temp_nbosons_array[end];
  }

  return convention_correction *
      (0.5 * domain->dimension * nbosons / beta + total_val);
}
