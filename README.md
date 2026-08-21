# LAMMPS with the middle-scheme NPT/NVT/NPH integrator

This is a LAMMPS fork that adds the **middle-scheme (BAOAB-type)**
integrator as new fix styles, together with a SHAKE mode compatible with
the middle splitting. The core implementation lives in
`fix_nh_middle.cpp` / `fix_nh_middle.h`, exposed through three user
entries:

- `fix nvt/mid` → `FixNVTMid`
- `fix nph/mid` → `FixNPHMid`
- `fix npt/mid` → `FixNPTMid`

All three share the same core class `FixNHMiddle`.

## Documentation

**`fix_nh_middle_manual.md`** — a full code-reader's manual for the
implementation (Chinese). It covers:

- **第零章** background: original `FixNH` / `fix nvt` / `fix nph` / `fix npt`
  functionality, keywords and calling conventions; a guide on how to read
  and modify `FixNH`-style integrators
- **第一章** the 10 key code changes: thermostat/barostat type and
  integration-order keywords, Langevin particle thermostat, Langevin box
  barostat, the 1/d isotropic correction, MPI sync of the barostat random
  term, the middle integration order, half-step position propagation, the
  `update_omega_dot()` barostat half-step, COM-random-kick removal, and
  the `zero_flag` / temperature-DOF coupling
- **第二章** input-file syntax and calling rules (incl. anisotropic /
  triclinic boxes, temperature-DOF handling, recommended templates)
- **第三章** complete test examples (Argon `npt/mid` + Langevin, q-SPC/Fw
  water)
- **第四章** reference on the **position-constraint implementations in the
  `RIGID` package** (`fix_shake` coordinate constraint in `middle` mode,
  `fix_rattle` position/velocity projection, `fix_rigid` rigid-body
  projection) for comparison with the middle scheme's C1/C2 constraints

## Key features

- **Integration order**: `integrator middle` (default recommended, more
  stable) or `integrator side` (the traditional OBABO-style order).
- **Thermostat / barostat types**: `langevin` (**default**; when no explicit
  relaxation time is given, thermostat damp = 500 fs and barostat damp =
  1000 fs) or `nh`, chosen independently via the `thermostat` and `barostat`
  keywords. A Langevin keyword takes one extra numeric argument: the
  relaxation time (fs).
- **Constraints**: the modified `fix shake` supports `middle yes` —
  coordinate (position) constraints only, applied in the middle-scheme
  slot, without an end-of-step velocity projection. Put `fix shake` BEFORE
  the `fix nvt/mid` / `fix npt/mid` command.
- **COM handling**: `zero 1` (default) removes the COM momentum from the
  random Langevin kick; `zero 0` restores the temperature DOF so no extra
  `compute_modify ... extra/dof 0` is needed.

## Quick examples

Middle NPT with Langevin thermostat + Langevin barostat:

```lammps
fix 1 all npt/mid iso (press args) temp (temp args) integrator middle \
  barostat langevin 1000.0 thermostat langevin 200.0
```

The pressure keywords `iso`, `aniso`, `tri`, `x`, `y`, `z`, `xy`, `xz`,
`yz` keep the same argument format as the original `fix npt`.

Middle NPT with constraints (SHAKE, coordinate-only, middle-compatible):

```lammps
fix SHAKE all shake 1e-4 100 0 b 1 a 1 middle yes
fix 1 all npt/mid iso (press args) temp (temp args) integrator middle \
  thermostat nh barostat nh
```

NPH middle barostat only (no thermostat):

```lammps
fix 1 all nph/mid barostat langevin 200.0 iso 0.986923 0.986923 200.0 \
  integrator middle
```

See `fix_nh_middle_manual.md` 第二章/第三章 for the full syntax and more
examples.
