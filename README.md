This is a new version of LAMMPS with middle-scheme NPT/NVT integrator.

The core new features are implemented in the file fix_nh_new.cpp and fix_nh_new.h

I have registered new fix style called fix npt/new, which is very similar to the original fix npt. More than that, you can choose the thermostat and barostat by adding keywords "barostat" and "thermostat" in fix command.

Now we support two types of thermostat and barostat, which are "langevin" and "nh". The default
for both is `nh`.

You can also choose to use different integration order by keywords "integrator", where you can choose "middle" or "side", and we recommand you to use middle integrator which is numerically more stable.

By default `zero 1` is used, which removes the COM momentum from the random Langevin kick. If you
set `zero 0`, this fix also restores the temperature DOF so you do not need an extra
`compute_modify ... extra/dof 0`.

There are also some flags added, but might be removed afterwards. Which actually do not matter too much, just set them to be 0


example:
if you want to use the middle integrator with langevin thermo/barostat then the fix command should be 

fix 1 all npt/new iso (press args) temp (temp args) integrator middle \
barostat langevin 1000.0 thermostat langevin 200.0

The pressure keywords `iso`, `aniso`, `tri`, `x`, `y`, `z`, `xy`, `xz`, and `yz` keep the same
argument format as the original `fix npt`.

If you choose `thermostat langevin` or `barostat langevin`, you must append one extra
numeric argument right after `langevin`. That value is the actual Langevin relaxation time used
for the thermal bath or pressure bath.

New fix shake now has been added, if you want to run NPT wiith constrian, simply add a fix shake command BEFORE the fix npt/new
and the fix shake command is different from the traditional ones, if you want to use shake compatible to middle integrator, you must add "middle yes" at the end of the fix command 

example:
fix SHAKE all shake 1e-4 100 0 b 1 a 1 middle yes
