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

#include "compute_temp_fixbackup.h"

#include "atom.h"
#include "error.h"
#include "fix.h"
#include "group.h"
#include "modify.h"
#include "update.h"
#include "force.h"
#include "domain.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempFixBackup::ComputeTempFixBackup(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), id_fix(nullptr), fix_snapshot(nullptr), v_backup_ptr(nullptr)
{
  if (narg != 4) error->all(FLERR, "Illegal compute temp/fixbackup command");

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;

  id_fix = utils::strdup(arg[3]);
  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

ComputeTempFixBackup::~ComputeTempFixBackup()
{
  delete[] id_fix;
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeTempFixBackup::init()
{
  fix_snapshot = modify->get_fix_by_id(id_fix);
  if (!fix_snapshot)
    error->all(FLERR, "Could not find fix {} for compute temp/fixbackup", id_fix);

  int dim = 0;
  auto *ptr = fix_snapshot->extract("v_backup", dim);
  if (!ptr || dim != 1)
    error->all(FLERR, "Fix {} does not expose a usable velocity backup", id_fix);

  v_backup_ptr = (double ***) ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeTempFixBackup::setup()
{
  dynamic = 0;
  if (dynamic_user || group->dynamic[igroup]) dynamic = 1;
  dof_compute();
}

/* ---------------------------------------------------------------------- */

void ComputeTempFixBackup::dof_compute()
{
  adjust_dof_fix();
  natoms_temp = group->count(igroup);
  dof = domain->dimension * natoms_temp;
  dof -= extra_dof + fix_dof;
  if (dof > 0.0)
    tfactor = force->mvv2e / (dof * force->boltz);
  else
    tfactor = 0.0;
}

/* ---------------------------------------------------------------------- */

double ComputeTempFixBackup::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  if (!v_backup_ptr || !*v_backup_ptr)
    error->all(FLERR, "Fix backup velocity snapshot is unavailable");

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double **v = *v_backup_ptr;

  double t = 0.0;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        t += (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) * rmass[i];
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        t += (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) * mass[type[i]];
  }

  MPI_Allreduce(&t, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  if (dynamic) dof_compute();
  if (dof < 0.0 && natoms_temp > 0.0)
    error->all(FLERR, "Temperature compute degrees of freedom < 0");
  scalar *= tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeTempFixBackup::compute_vector()
{
  invoked_vector = update->ntimestep;

  if (!v_backup_ptr || !*v_backup_ptr)
    error->all(FLERR, "Fix backup velocity snapshot is unavailable");

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double **v = *v_backup_ptr;

  double massone, t[6];
  for (int i = 0; i < 6; i++) t[i] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass)
        massone = rmass[i];
      else
        massone = mass[type[i]];
      t[0] += massone * v[i][0] * v[i][0];
      t[1] += massone * v[i][1] * v[i][1];
      t[2] += massone * v[i][2] * v[i][2];
      t[3] += massone * v[i][0] * v[i][1];
      t[4] += massone * v[i][0] * v[i][2];
      t[5] += massone * v[i][1] * v[i][2];
    }

  MPI_Allreduce(t, vector, 6, MPI_DOUBLE, MPI_SUM, world);
  for (int i = 0; i < 6; i++) vector[i] *= force->mvv2e;
}
