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

#include "fix_pimd_langevin_bosonic_new.h"

#include "bosonic_exchange_new.h"

#include "atom.h"
#include "error.h"
#include "memory.h"
#include "universe.h"
#include "random_mars.h"

#include <cstring>
#include <string>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixPIMDBLangevinNew::FixPIMDBLangevinNew(LAMMPS *lmp, int narg, char **arg) :
    FixPIMDLangevin(lmp, narg, filtered_args = filter_args(narg, arg)), filtered_narg(narg),
    nbosons(atom->nlocal)
{
  printf("Initializing FixPIMDBLangevinNew with %d bosons\n", nbosons);

  // Initialize parameters with default values
  BosonicExchangeNew::RejectMethod reject_method = BosonicExchangeNew::REJECT_PROB; // Default to PROB as in python example? Or allow setting.
  double cutoff = 10000.0;
  double prob_cutoff = 1e-7;
  double prob_recover = 0.1;

  // Parse new arguments before creating BosonicExchangeNew
  // Arguments are in 'arg'. FixPIMDLangevin consumes standard args.
  // We need to look for custom args here.
  // However, FixPIMDLangevin constructor already parsed everything? 
  // No, FixPIMDLangevin constructor parses standard args. Unknown args might cause error if not careful?
  // FixPIMDLangevin constructor: iterates loop. If unknown, error.
  // BUT we filtered arguments passed to Parent!
  // 'filter_args' removes "esynch". We should also remove our new args so parent doesn't error.
  
  // Note: f_tag_order allocation moved down
  
  // Parse arguments again to find our specific settings
  for (int i = 3; i < narg; i++) {
      if (strcmp(arg[i], "reject_method") == 0) {
          if (strcmp(arg[i+1], "distance") == 0) reject_method = BosonicExchangeNew::REJECT_DISTANCE;
          else if (strcmp(arg[i+1], "prob") == 0) reject_method = BosonicExchangeNew::REJECT_PROB;
          else reject_method = BosonicExchangeNew::REJECT_NONE;
          i++;
      } else if (strcmp(arg[i], "cutoff") == 0) {
          cutoff = utils::numeric(FLERR, arg[i+1], false, lmp);
          i++;
      } else if (strcmp(arg[i], "prob_cutoff") == 0) {
          prob_cutoff = utils::numeric(FLERR, arg[i+1], false, lmp);
          i++;
      } else if (strcmp(arg[i], "prob_recover") == 0) {
          prob_recover = utils::numeric(FLERR, arg[i+1], false, lmp);
          i++;
      }
  }

  // Determine if random generator is needed (for recover)
  // FixPIMDLangevin has 'random' member (RanMars).
  // We can pass it.
  
  bosonic_exchange = new BosonicExchangeNew(lmp, atom->nlocal, np, universe->me, true, false, random);
  bosonic_exchange->set_rejection_params(reject_method, cutoff, prob_cutoff, prob_recover);
  
  synch_energies = true;

  // Loop over the arguments i++ to check standard params (copied from original)
  for (int i = 3; i < narg - 1; i++) {
    if ((strcmp(arg[i], "method") == 0) && (strcmp(arg[i + 1], "pimd") != 0)) {
      error->universe_all(FLERR, "Method not supported in fix pimdb/langevin/new; only method PIMD");
    } else if (strcmp(arg[i], "esynch") == 0) {
      if (strcmp(arg[i + 1], "yes") == 0) {
        synch_energies = true;
      } else if (strcmp(arg[i + 1], "no") == 0) {
        synch_energies = false;
      } else {
        error->universe_all(FLERR, "The esynch parameter can only receive yes or no!");
      }
    }
  }

  if (fmmode != PHYSICAL) {
    error->universe_all(
        FLERR,
        "The only available fmmode for pimdb is physical, please remove the fmmode keyword.");
  }
  if (ensemble != NVE && ensemble != NVT) {
    error->universe_all(FLERR,
                        "The only available ensembles for pimdb are nve and nvt, please choose one "
                        "of these ensembles.");
  }

  method = PIMD;
  size_vector = 6;
  memory->create(f_tag_order, nbosons, 3, "FixPIMDBLangevinNew:f_tag_order");

  if (cmode != SINGLE_PROC)
    error->universe_all(FLERR,
                        fmt::format("Fix {} only supports a single processor per bead", style));
}

/* ---------------------------------------------------------------------- */

FixPIMDBLangevinNew::~FixPIMDBLangevinNew()
{
  memory->destroy(f_tag_order);
  for (int i = 0; i < filtered_narg; ++i) delete[] filtered_args[i];
  delete[] filtered_args;
  delete bosonic_exchange;
}

/* ---------------------------------------------------------------------- */

char **FixPIMDBLangevinNew::filter_args(int narg, char **arg)
{
  filtered_narg = 0; // Will count later
  
  // We need to count valid args for parent
  int valid_count = 0;
  for (int i = 0; i < narg; i++) {
     bool is_custom = false;
     if (strcmp(arg[i], "esynch") == 0 ||
         strcmp(arg[i], "reject_method") == 0 ||
         strcmp(arg[i], "cutoff") == 0 ||
         strcmp(arg[i], "prob_cutoff") == 0 ||
         strcmp(arg[i], "prob_recover") == 0) {
         is_custom = true;
         // Skip value
         // Check if next arg exists and isn't a keyword? 
         // Most usage implies value follows.
         // But "esynch" takes "yes/no".
     }
     
     // Actually, we need to construct a new array excluding our params
     // because parent constructor iterates and might error on unknown params.
     // But wait, parent constructor logic:
     // for (int i = 3; i < narg - 1; i += 2) ...
     // It expects key-value pairs?
     // FixPIMDLangevin.cpp :
     // for (int i = 3; i < narg - 1; i += 2) { if (strcmp(arg[i], ...) ... else error }
     // So we MUST filter out our keys and their values.
  }
  
  // First pass: count non-custom args
  // Assume basic 3 args (fix ID group style) are always kept?
  // Fix constructor takes narg, arg. 
  // We allocate conservative size.
  
  char **filtered_args = new char *[narg];
  int count = 0;
  
  for (int i = 0; i < narg; i++) {
     if (i < 3) {
         filtered_args[count++] = utils::strdup(arg[i]);
         continue;
     }
     
     if (strcmp(arg[i], "esynch") == 0 || 
         strcmp(arg[i], "reject_method") == 0 || 
         strcmp(arg[i], "cutoff") == 0 || 
         strcmp(arg[i], "prob_cutoff") == 0 || 
         strcmp(arg[i], "prob_recover") == 0) {
         // Skip this key and the next value
         i++; 
     } else {
         filtered_args[count++] = utils::strdup(arg[i]);
     }
  }
  
  filtered_narg = count;
  return filtered_args;
}

/* ---------------------------------------------------------------------- */

void FixPIMDBLangevinNew::prepare_coordinates()
{
  inter_replica_comm(atom->x);
  double ff = fbond * atom->mass[atom->type[0]];
  int nlocal = atom->nlocal;
  double *me_bead_positions = *(atom->x);
  double *last_bead_positions = &bufsortedall[x_last * nlocal][0];
  double *next_bead_positions = &bufsortedall[x_next * nlocal][0];

  bosonic_exchange->prepare_with_coordinates(me_bead_positions, last_bead_positions,
                                             next_bead_positions, beta_np, ff);
}

/* ---------------------------------------------------------------------- */

void FixPIMDBLangevinNew::spring_force()
{

  for (int i = 0; i < nbosons; i++) {
    f_tag_order[i][0] = 0.0;
    f_tag_order[i][1] = 0.0;
    f_tag_order[i][2] = 0.0;
  }
  bosonic_exchange->spring_force(f_tag_order);

  double **f = atom->f;
  tagint *tag = atom->tag;
  for (int i = 0; i < nbosons; i++) {
    f[i][0] += f_tag_order[tag[i] - 1][0];
    f[i][1] += f_tag_order[tag[i] - 1][1];
    f[i][2] += f_tag_order[tag[i] - 1][2];
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDBLangevinNew::compute_spring_energy()
{
  se_bead = bosonic_exchange->get_bead_spring_energy();

  if (synch_energies) {
    MPI_Allreduce(&se_bead, &total_spring_energy, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    total_spring_energy /= universe->procs_per_world[universe->iworld];
  } else {
    total_spring_energy = 0;
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDBLangevinNew::compute_t_prim()
{
  if (synch_energies) {
    double prim = bosonic_exchange->prim_estimator();
    MPI_Allreduce(&prim, &t_prim, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  } else {
    t_prim = bosonic_exchange->prim_estimator();
  }
}

/* ---------------------------------------------------------------------- */

double FixPIMDBLangevinNew::compute_vector(int n)
{
  if (0 <= n && n < 6) {
    return FixPIMDLangevin::compute_vector(n);
  } else {
    error->universe_all(FLERR, "Fix only has 6 outputs!");
  }
  return 0.0;
}
