This is a new version of LAMMPS with middle-scheme NPT/NVT integrator.

The core new features are implemented in the file fix_nh_new.cpp and fix_nh_new.h

I have registered new fix style called fix npt/new, which is very similar to the original fix npt. More than that, you can choose the thermostat and barostat by adding keywords "barostat" and "thermostat" in fix command.

Now we support two types of thermostat and barostat, which are "langevin" and "nh".

You can also choose to use different integration order by keywords "integrator", where you can choose "middle" or "side", and we recommand you to use middle integrator which is numerically more stable.

To remove the COM momentum when using langevin thermostat, you can use "zero_flag 0".

There are also some flags added, but might be removed afterwards. Which actually do not matter too much, just set them to be 0


example:
$$
\begin{align}
 & \text{fix 1 all} \textcolor{red}{\text{ middle/new }} \& \\
 & \text{temp 299.5 299.5 432.9 }\& \\
 & \text{iso 705.423 705.423 4329.0 1000.0 2.4 }\& \\
 & \textcolor{red}{\text{thermostat langevin }}\& \\
 & \textcolor{red}{\text{barostat nh }} \& \\
 & \text{big\_mass 1 big\_update 1}
\end{align}
$$
