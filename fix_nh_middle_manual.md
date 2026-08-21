# `fix_nh_middle` Manual

本文档面向代码读者，说明 `fix_nh_middle` 相对于原版 `FixNH` 的关键改动，并给出 `fix nvt/mid`、`fix nph/mid`、`fix npt/mid` 的输入文件写法。
这三个用户入口分别对应：

- `fix nvt/mid` -> `FixNVTMid`
- `fix nph/mid` -> `FixNPHMid`
- `fix npt/mid` -> `FixNPTMid`

三者的核心实现均位于 `FixNHMiddle`。

全文分为五章：

0. 第零章说明原版 `FixNH` 与原版 `fix npt` 的基本功能和调用方式
1. 第一章说明程序实现中的 10 个关键改动
2. 第二章说明输入文件的写法与调用规则
3. 第三章给出一个完整的简化测试样例
4. 第四章整理 RIGID 包中的位置约束实现（`fix_shake` / `fix_rattle` /
   `fix_rigid`），供与 middle 方案的 C1/C2 约束对照

若读者当前只关心如何在 LAMMPS 输入脚本中调用 `fix nvt/mid`、`fix nph/mid` 或 `fix npt/mid`，可直接跳转到第二章。
若读者更关心本实现的开发思路，或者想学习如何从原版 `FixNH` 入手阅读和修改这类积分器代码，可直接阅读第 0.5 节“阅读和修改 `FixNH` 的开发思路”。

## 第零章：原版 `FixNH` 与原版 `fix nvt` / `fix nph` / `fix npt` 的对照说明

本章的目的不是完整重述 LAMMPS 手册，而是给出一个足够简洁的对照背景。
后文中 `fix_nh_middle` 的新增功能和输入方式，均是在这一基础上扩展而来。

### 0.1 原版 `FixNH` 支持的主要功能

原版 `FixNH` 是 LAMMPS 中 NVT、NPH、NPT 一类 Nose-Hoover 积分器的共同基类。
从类定义可以直接看出，它内部已经包含以下几类核心能力：

- 温度控制变量与 Nose-Hoover chain
- 压力控制变量与 barostat
- 各向同性、各向异性与三斜盒子的压力控制
- 压力分量耦合
- 盒子 remap 与速度缩放
- MTK 修正项
- rRESPA 支持
- restart、thermo 输出与 `fix_modify` 支持

类接口如下：

```cpp
class FixNH : public Fix {
 public:
  FixNH(class LAMMPS *, int, char **);
  ~FixNH() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void initial_integrate_respa(int, int, int) override;
  void pre_force_respa(int, int, int) override;
  void final_integrate_respa(int, int) override;
  void pre_exchange() override;
  double compute_scalar() override;
  double compute_vector(int) override;
  std::string get_thermo_colname(int) override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  int modify_param(int, char **) override;
  void reset_target(double) override;
  void reset_dt() override;
  void *extract(const char *, int &) override;
  double memory_usage() override;
```

其内部成员中，与 NPT 模拟最直接相关的变量包括：

```cpp
int tstat_flag;
int pstat_flag;

int pstyle, pcouple, allremap;
int p_flag[6];
double p_start[6], p_stop[6];
double p_freq[6], p_target[6];
double omega[6], omega_dot[6];
double omega_mass[6];
double p_current[6];

double *eta, *eta_dot;
double *etap, *etap_dot;

int mtk_flag;
int pdim;
double p_hydro;
double mtk_term1, mtk_term2;
```

其中，`omega`、`omega_dot` 和 `omega_mass` 是理解 NPT 算法时最关键的一组 barostat 变量：

- `omega[6]`：表示盒子自由度的广义坐标，最多包含 6 个压力/盒子分量，前三个对应 `x`、`y`、`z` 方向的正交盒长涨缩，后三个对应三斜盒子的 `yz`、`xz`、`xy` tilt 分量。
- `omega_dot[6]`：表示这些盒子自由度的广义速度，也就是盒子涨缩或 tilt 演化的速率。NPT 积分中，压力偏离目标压力时并不是直接改原子坐标，而是先更新 `omega_dot`，再由 `remap()` 等步骤用它推进盒子矩阵和原子坐标。
- `omega_mass[6]`：表示每个盒子自由度对应的有效 barostat 质量，决定 `omega_dot` 对压力偏差响应的快慢；它由压力阻尼时间、目标温度和体系大小等量决定。

因此，在 NPT 模拟中，`omega_dot` 扮演的是“盒子动量/盒子速度”的角色。当前内压 `p_current` 与目标压力 `p_target` 的差值会通过 `nh_omega_dot()` 或本文实现中的 `update_omega_dot()` 转化为对 `omega_dot` 的更新；随后 `omega_dot` 决定盒子体积、盒长或 tilt 因子如何随时间变化。也就是说，NPT 控压的核心并不是直接把压力强行设为目标值，而是通过 barostat 自由度 `omega_dot` 动态调节模拟盒子，使体系在统计意义上采样目标压力。

这些成员表明，原版 `FixNH` 已经能够处理：

- 粒子热浴变量
- barostat 变量
- 各压力分量目标值
- 压力耦合方式
- MTK 修正

换言之，`fix_nh_middle` 的实现并不是重新发明一套 NPT 数据结构，而是在原版 `FixNH` 已有框架上增加新的热浴类型与新的积分顺序。

### 0.2 原版 `FixNH` 支持的主要输入关键字

原版 `FixNH` 在构造函数中解析温度、压力和 barostat 相关关键字。
其关键解析代码如下：

```cpp
while (iarg < narg) {
  if (strcmp(arg[iarg],"temp") == 0) {
    ...
  } else if (strcmp(arg[iarg],"iso") == 0) {
    ...
  } else if (strcmp(arg[iarg],"aniso") == 0) {
    ...
  } else if (strcmp(arg[iarg],"tri") == 0) {
    ...
  } else if (strcmp(arg[iarg],"x") == 0) {
    ...
  } else if (strcmp(arg[iarg],"y") == 0) {
    ...
  } else if (strcmp(arg[iarg],"z") == 0) {
    ...
  } else if (strcmp(arg[iarg],"yz") == 0) {
    ...
  } else if (strcmp(arg[iarg],"xz") == 0) {
    ...
  } else if (strcmp(arg[iarg],"xy") == 0) {
    ...
  } else if (strcmp(arg[iarg],"couple") == 0) {
    ...
  } else if (strcmp(arg[iarg],"drag") == 0) {
    ...
```

对 NPT 代码读者而言，最需要先看清楚的是压力控制关键字，因为它们决定 `p_flag`、`pcouple`、`pstyle`，进而决定 `omega_dot` 有几个有效自由度以及 `pressure->compute_*()` 后如何解释压力张量。

原版 `FixNH` 中最常见的三种压力控制写法是：

- `iso Pstart Pstop Pdamp`：各向同性控压。源码中会设置 `pcouple = XYZ`，并打开 `x/y/z` 三个正交压力分量。在三维中，这等价于三个方向共享同一个目标压力、同一个当前平均压力和一个各向同性体积涨缩模式；在二维中只保留 `x/y`。最终 `pstyle` 会被判定为 `ISO`。
- `aniso Pstart Pstop Pdamp`：正交各向异性控压。源码中会设置 `pcouple = NONE`，并打开 `x/y/z` 三个正交压力分量。三个方向使用相同的输入起止压力和阻尼时间初始化，但盒长可以独立涨缩，当前压力也分别取 `Pxx`、`Pyy`、`Pzz`。最终通常被判定为 `ANISO`。
- `tri Pstart Pstop Pdamp`：三斜盒子控压。源码中会设置 `pcouple = NONE`，打开 `x/y/z` 以及 `xy/xz/yz` tilt 压力分量，并关闭相关 tilt scaling 标志。它允许盒子形状随剪切分量一起演化，最终被判定为 `TRICLINIC`。

也可以不用 `iso/aniso/tri`，而是逐项写：

- `x/y/z Pstart Pstop Pdamp`：分别打开正交压力张量的对角分量。
- `xy/xz/yz Pstart Pstop Pdamp`：分别打开三斜盒子的 off-diagonal tilt 分量；这些关键字要求三斜盒子。

这里最容易混淆的是 `couple`。它不是单纯把几个输入压力写在一起，而是告诉 `FixNH` 哪些正交方向应该作为一个耦合的压力子空间处理。源码中的 `couple()` 会按 `pcouple` 改写 `p_current`：

- `couple none`：不耦合。`p_current[0:2]` 分别使用压力张量中的 `Pxx`、`Pyy`、`Pzz`，各方向独立控压。
- `couple xyz`：三维中耦合 `x/y/z`。`p_current[0] = p_current[1] = p_current[2] = (Pxx + Pyy + Pzz)/3`，三个方向使用同一个平均压力，目标压力和阻尼时间也必须一致。
- `couple xy`：耦合 `x/y`，使用 `(Pxx + Pyy)/2`，`z` 方向保持独立。
- `couple yz`：耦合 `y/z`，使用 `(Pyy + Pzz)/2`，`x` 方向保持独立。
- `couple xz`：耦合 `x/z`，使用 `(Pxx + Pzz)/2`，`y` 方向保持独立。

因此，可以把 `couple` 理解为“哪些盒长方向要共享当前压力反馈并一起涨缩”。`iso` 是最强的耦合写法，直接等价于三维 `couple xyz` 的各向同性模式；`aniso` 和 `tri` 默认 `couple none`，但用户逐项指定 `x/y/z` 时也可以显式加 `couple xy/yz/xz/xyz` 来获得部分耦合。

压力之外，原版 `FixNH` 中还需要知道几个温度和链变量关键字：

- `temp Tstart Tstop Tdamp`：开启温度控制，设置目标温度从 `Tstart` 到 `Tstop` 的变化范围，以及 Nose-Hoover 温度热浴的阻尼时间 `Tdamp`。
- `tchain N`：设置粒子温度 Nose-Hoover chain 的长度，影响 `nhc_temp_integrate()` 中温度热浴变量的个数。
- `pchain N`：设置 barostat Nose-Hoover chain 的长度，影响 `nhc_press_integrate()` 中压浴链变量的个数；`pchain 0` 表示不使用压浴链。
- `mtk yes/no`：控制是否启用 MTK 修正项。该项会影响 `nh_omega_dot()` 中的 `mtk_term1` 和 `mtk_term2`，也就是盒子动量更新和粒子速度缩放中的相空间测度修正。

其他高级关键字例如 `drag`、`ptemp`、`dilate`、`nreset`、`tloop`、`ploop`、`fixedpoint`、`flip`、`scalexy/scalexz/scaleyz` 等仍由原版 `FixNH` 解析，但不是理解本文 `fix_nh_middle` 改动的主线。

后文 `fix_nh_middle` 新增的关键字是：

- `thermostat nh|langevin [Tdamp_lan]`：选择粒子温度热浴类型；`nh` 使用原版 Nose-Hoover chain，`langevin` 使用本文新增的粒子速度 Langevin O-step。
- `barostat nh|langevin [Pdamp_lan]`：选择 barostat 自由度的热浴类型；`nh` 使用原版压浴链，`langevin` 使用本文新增的盒子自由度 Langevin O-step。
- `integrator side|middle`：选择时间分裂顺序；`side` 走接近原版 `FixNH` 的 side 顺序，`middle` 走本文新增的 middle 顺序。
- `seed integer`：设置 Langevin 随机数种子，只在启用粒子或压浴 Langevin 时影响轨迹。
- `zero yes/no`：控制粒子 Langevin 随机 kick 是否去除质心净冲量；在 `thermostat langevin` 路径下还联动内部温度 compute 的 `extra/dof` 设置。

均不属于原版 `FixNH` 的原生输入关键字。

### 0.3 原版 `fix nvt`、`fix nph`、`fix npt` 的角色

原版 `fix nvt`、`fix nph`、`fix npt` 都不是独立重写的一套算法，而是对 `FixNH` 的薄封装。
其类定义分别为：

```cpp
class FixNVT : public FixNH {
 public:
  FixNVT(class LAMMPS *, int, char **);
};

class FixNPH : public FixNH {
 public:
  FixNPH(class LAMMPS *, int, char **);
};

class FixNPT : public FixNH {
 public:
  FixNPT(class LAMMPS *, int, char **);
};
```

三者的职责分工如下：

- `FixNVT`
  - 要求必须有温度控制
  - 禁止压力控制
  - 自动创建内部 `compute temp`

- `FixNPH`
  - 禁止温度控制
  - 要求必须有压力控制
  - 自动创建内部 `compute temp`
  - 自动创建内部 `compute pressure`

- `FixNPT`
  - 要求必须同时有温度控制和压力控制
  - 自动创建内部 `compute temp`
  - 自动创建内部 `compute pressure`

对应代码如下。

原版 `fix nvt`：

```cpp
FixNVT::FixNVT(LAMMPS *lmp, int narg, char **arg) : FixNH(lmp, narg, arg)
{
  if (!tstat_flag) error->all(FLERR, "Temperature control must be used with fix nvt");
  if (pstat_flag) error->all(FLERR, "Pressure control can not be used with fix nvt");

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp", id_temp, group->names[igroup]));
  tcomputeflag = 1;
}
```

原版 `fix nph`：

```cpp
FixNPH::FixNPH(LAMMPS *lmp, int narg, char **arg) : FixNH(lmp, narg, arg)
{
  if (tstat_flag) error->all(FLERR, "Temperature control can not be used with fix nph");
  if (!pstat_flag) error->all(FLERR, "Pressure control must be used with fix nph");

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp", id_temp));
  tcomputeflag = 1;

  id_press = utils::strdup(std::string(id) + "_press");
  modify->add_compute(fmt::format("{} all pressure {}", id_press, id_temp));
  pcomputeflag = 1;
}
```

原版 `fix npt`：

```cpp
FixNPT::FixNPT(LAMMPS *lmp, int narg, char **arg) : FixNH(lmp, narg, arg)
{
  if (!tstat_flag) error->all(FLERR, "Temperature control must be used with fix npt");
  if (!pstat_flag) error->all(FLERR, "Pressure control must be used with fix npt");

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp", id_temp));
  tcomputeflag = 1;

  id_press = utils::strdup(std::string(id) + "_press");
  modify->add_compute(fmt::format("{} all pressure {}", id_press, id_temp));
  pcomputeflag = 1;
}
```

这意味着原版这三类 fix 的本质都是：

- 由 `FixNH` 负责具体积分与参数解析
- 由外层 wrapper 决定当前是 NVT、NPH 还是 NPT，并自动补齐内部 compute

这一点与本文实现的 `FixNVTMid`、`FixNPHMid`、`FixNPTMid` 完全对应，因此后文第二章中的 `mid` 版本调用格式会与原版三类 fix 高度相似。

### 0.4 原版 `fix nvt` / `fix nph` / `fix npt` 的基本调用方式

原版 `fix nvt` 的最常见写法为：

```lammps
fix 1 all nvt temp 300.0 300.0 200.0
```

这条命令只包含温度控制，不包含任何压力关键字。
逐项解释为：

- `fix`：LAMMPS 中定义一个 fix 的命令。
- `1`：fix 的 ID，后续可用 `unfix 1`、`fix_modify 1 ...` 等命令引用它。
- `all`：作用的 group-ID，表示这个 fix 作用于 `all` 组中的原子。
- `nvt`：fix style，表示使用原版 Nose-Hoover NVT 积分器。
- `temp`：温度控制关键字，表示后面三个数值是温度控制参数。
- 第一个 `300.0`：起始目标温度 `Tstart`。
- 第二个 `300.0`：结束目标温度 `Tstop`；这里与 `Tstart` 相同，表示恒温 300 K。
- `200.0`：温度阻尼时间 `Tdamp`，控制 Nose-Hoover 温度热浴响应的时间尺度。

原版 `fix nph` 的最常见写法为：

```lammps
fix 1 all nph iso 1.0 1.0 1000.0
```

这条命令只包含压力控制，不包含 `temp` 关键字。
逐项解释为：

- `fix`：定义一个 fix 的命令。
- `1`：fix 的 ID。
- `all`：作用的 group-ID。
- `nph`：fix style，表示使用原版 Nose-Hoover NPH 积分器，只控压、不控温。
- `iso`：各向同性压力控制关键字，表示盒子在受控方向上按同一个压力目标涨缩。
- 第一个 `1.0`：起始目标压力 `Pstart`。
- 第二个 `1.0`：结束目标压力 `Pstop`；这里与 `Pstart` 相同，表示目标压力保持不变。
- `1000.0`：压力阻尼时间 `Pdamp`，控制 barostat 响应的时间尺度。

原版 `fix npt` 的最常见调用写法为：

```lammps
fix 1 all npt temp 300.0 300.0 200.0 iso 1.0 1.0 1000.0
```

其中：

- `temp 300.0 300.0 200.0` 指定温度起点、终点和阻尼时间
- `iso 1.0 1.0 1000.0` 指定各向同性目标压强及其阻尼时间
- `fix`、`1`、`all` 的含义与上面相同，分别表示定义 fix、fix ID 和作用组。
- `npt` 是 fix style，表示同时使用 Nose-Hoover 温度热浴和 barostat。
- `temp` 后的第一个 `300.0` 是 `Tstart`，第二个 `300.0` 是 `Tstop`，`200.0` 是 `Tdamp`。
- `iso` 后的第一个 `1.0` 是 `Pstart`，第二个 `1.0` 是 `Pstop`，`1000.0` 是 `Pdamp`。

若采用各向异性或三斜盒子，也可以写成：

```lammps
fix 1 all npt temp 300.0 300.0 200.0 aniso 1.0 1.0 1000.0
```

这里前半段 `fix 1 all npt temp 300.0 300.0 200.0` 的含义与上一个 NPT 示例相同。
后半段 `aniso 1.0 1.0 1000.0` 表示各向异性压力控制：第一个 `1.0` 是 `Pstart`，第二个 `1.0` 是 `Pstop`，`1000.0` 是 `Pdamp`；与 `iso` 不同，`aniso` 允许正交盒长方向独立涨缩。

或

```lammps
fix 1 all npt temp 300.0 300.0 200.0 tri 1.0 1.0 1000.0
```

这里 `tri 1.0 1.0 1000.0` 表示三斜盒子压力控制：第一个 `1.0` 是 `Pstart`，第二个 `1.0` 是 `Pstop`，`1000.0` 是 `Pdamp`；除了正交方向外，`tri` 还会包含 `xy`、`xz`、`yz` 三个 tilt 压力分量。

也可以逐个分量指定，例如：

```lammps
fix 1 all npt temp 300.0 300.0 200.0 \
               x 1.0 1.0 1000.0 \
               y 1.0 1.0 1000.0 \
               z 1.0 1.0 1000.0 \
               couple none
```

这条命令逐项解释为：

- `fix 1 all npt`：定义 ID 为 `1`、作用于 `all` 组的原版 NPT fix。
- `temp 300.0 300.0 200.0`：温度目标从 `300.0` 到 `300.0`，温度阻尼时间为 `200.0`。
- `x 1.0 1.0 1000.0`：控制 `x` 方向压力分量，`Pstart = 1.0`，`Pstop = 1.0`，`Pdamp = 1000.0`。
- `y 1.0 1.0 1000.0`：控制 `y` 方向压力分量，三个数值含义同上。
- `z 1.0 1.0 1000.0`：控制 `z` 方向压力分量，三个数值含义同上；二维模拟中不能使用 `z` 压力控制。
- `couple none`：不把 `x`、`y`、`z` 压力方向耦合在一起，因此三个方向可以按各自的 barostat 变量独立涨缩。

这些写法的共同特点是：

1. 输入语法完全围绕 `temp + 压力关键字` 组织
2. 热浴类型默认就是 Nose-Hoover chain
3. 用户无法在原版 `fix nvt`、`fix nph`、`fix npt` 中额外指定 `thermostat`、`barostat`、`integrator`、`seed` 或 `zero`

### 0.5 阅读和修改 `FixNH` 的开发思路

本文档不仅说明当前代码做了什么，也希望帮助读者学会如何阅读和修改这一类 LAMMPS 积分器代码。
拿到原版 `fix_nh.cpp` 后，最重要的第一步不是直接改某个热浴函数，而是先理清 LAMMPS 的时间推进主循环，以及 `FixNH` 在这个主循环中被调用的位置。

LAMMPS 的 Velocity-Verlet 主循环由 `verlet.cpp` 控制。论文第二章“Verlet主循环与积分器调用位置”中给出了对应伪代码，可简化为：

```text
loop over timesteps:
  ev_set()
  fix->initial_integrate()
  fix->post_integrate()

  if neighbor list needs rebuild:
    fix->pre_exchange()
    domain->pbc()
    domain->reset_box()
    comm->setup()
    comm->exchange()
    comm->borders()
    fix->pre_neighbor()
    neighbor->build()
    fix->post_neighbor()
  else:
    comm->forward_comm()

  force_clear()
  fix->pre_force()
  pair/bond/angle/dihedral/improper/kspace force calculations
  fix->pre_reverse()
  comm->reverse_comm()
  fix->post_force()

  fix->final_integrate()
  fix->end_of_step()
  output->write()
```

因此，`fix` 类中真正承接积分流程的两个核心入口是：

- `initial_integrate()`：在力计算之前调用，通常负责前半步速度更新、位置更新、盒子前半步 remap，以及需要放在受力计算前完成的温度/压力更新。
- `final_integrate()`：在力计算和反向通信之后调用，通常负责后半步速度更新、重新计算温度压力、完成热浴和压浴的后半步更新。

对原版 `FixNH` 来说，阅读顺序应当从这两个函数开始。其主流程可以概括为：

```text
FixNH::initial_integrate:
  nhc_press_integrate()
  compute_temp_target()
  nhc_temp_integrate()
  compute T, P
  compute_press_target()
  nh_omega_dot()
  nh_v_press()
  nve_v()
  remap()
  nve_x()
  remap()
  kspace->setup()

Force evaluation happens in verlet.cpp

FixNH::final_integrate:
  nve_v()
  nh_v_press()
  compute T, P
  nh_omega_dot()
  nhc_temp_integrate()
  nhc_press_integrate()
```

看懂这两个函数后，下一步是把每个函数和理论中的演化算子对应起来。一个实用的对应关系是：

- `nve_v()`：粒子速度的力推进，对应 $\mathcal{L}_{v_t}$。
- `nve_x()`：粒子位置推进，对应 $\mathcal{L}_{x_t}$。
- `remap()`：盒子矩阵和坐标随 barostat 变量更新，对应盒子演化和坐标缩放相关算子。
- `nh_v_press()`：barostat 对粒子速度的缩放，对应 $\mathcal{L}_{v_r}$。
- `nh_omega_dot()`：根据当前压力和目标压力更新盒子速度变量 `omega_dot`，对应 $\mathcal{L}_{v_{\varepsilon}}$。
- `nhc_temp_integrate()`：Nose-Hoover chain 温度热浴更新，对应 $\mathcal{L}_{T}^{\mathrm{NHC}}$。
- `nhc_press_integrate()`：Nose-Hoover chain 压浴更新，对应 $\mathcal{L}_{B}^{\mathrm{NHC}}$，并与盒子动量变量的热化有关。
- `compute_temp_target()` 和 `compute_press_target()`：更新当前时间步的目标温度和目标压力。
- `temperature->compute_*()` 与 `pressure->compute_*()`：重新计算当前温度、压力和动能张量，为后续 `nh_omega_dot()` 或 `update_omega_dot()` 提供输入。

有了这张对应关系之后，修改代码的思路就比较清楚：如果要改变积分顺序，就重排 `initial_integrate()` 和 `final_integrate()` 中这些算子的调用位置；如果要改变某个物理过程，就替换或改写对应的算子函数。例如，本文实现中的主要改动可以按以下方式理解：

- 要把 side 顺序改为 middle 顺序，就需要重写 `initial_integrate()` / `final_integrate()` 中的算子排列，并新增 `nve_x_half()` 来实现半步位置推进。
- 要把温度热浴从 Nose-Hoover chain 换成 Langevin，就不能只改参数解析，而要在原来调用 `nhc_temp_integrate()` 的位置切换到 `langevin_temp()`。
- 要把压浴热化从 Nose-Hoover chain 换成 Langevin，就要在原来调用 `nhc_press_integrate()` 的位置切换到 `langevin_press()`，并处理 `omega_dot` 这类全局盒子变量的 MPI 同步。
- 要修改盒子动量更新，就应当看原版 `nh_omega_dot()`。本文将对应功能改写为 `update_omega_dot()`，仍然执行更新 `omega_dot` 的职责，但根据当前 middle 路径和自由度定义调整了 MTK 项的处理。
- 要处理 NPT 中的全局变量一致性，就要特别关注 `omega_dot`、盒子矩阵和 `remap()`。粒子变量是 per-atom 的，可以本地更新；盒子自由度是全局变量，随机项或确定性更新必须保证所有 MPI rank 一致。

因此，阅读 `fix_nh_middle` 时可以采用同样的路线：先看 `initial_integrate()` 和 `final_integrate()` 的总体顺序，再分别进入 `integrate_temp_thermostat()`、`integrate_press_thermostat()`、`langevin_temp()`、`langevin_press()`、`update_omega_dot()` 和 `nve_x_half()`。这样做的好处是，读者看到的不是一堆孤立函数，而是一套可以按算子拆分理解、也可以按算子位置进行修改的 NPT 积分框架。

### 0.6 本文实现与原版三类 fix 的直接对比

为了便于后文阅读，可以先给出最直接的对比。

原版 `fix nvt` 的典型写法是：

```lammps
fix 1 all nvt temp 300.0 300.0 200.0
```

而对应的 `mid` 版本可以写成：

```lammps
fix 1 all nvt/mid temp 300.0 300.0 200.0 \
               integrator middle \
               thermostat langevin 200.0 \
               seed 123456 \
               zero yes
```

原版 `fix nph` 的典型写法是：

```lammps
fix 1 all nph iso 1.0 1.0 1000.0
```

而对应的 `mid` 版本可以写成：

```lammps
fix 1 all nph/mid iso 1.0 1.0 1000.0 \
               integrator middle \
               barostat langevin 1000.0 \
               seed 123456
```

原版 `fix npt` 的典型写法是：

```lammps
fix 1 all npt temp 300.0 300.0 200.0 iso 1.0 1.0 1000.0
```

而本文实现 `fix npt/mid` 的典型写法则是：

```lammps
fix 1 all npt/mid temp 300.0 300.0 200.0 \
               iso 1.0 1.0 1000.0 \
               integrator middle \
               thermostat langevin 200.0 \
               barostat langevin 1000.0 \
               seed 123456 \
               zero yes
```

二者的共同点在于：

- `nvt` 与 `nvt/mid` 都以 `temp` 为基本骨架
- `nph` 与 `nph/mid` 都以压力关键字为基本骨架
- `npt` 与 `npt/mid` 都以 `temp + 压力关键字` 为基本骨架
- 三类 wrapper 都自动创建内部 compute

二者的区别在于：

- 原版 `fix nvt`、`fix nph`、`fix npt` 仅支持原生 Nose-Hoover 风格实现
- `fix nvt/mid`、`fix nph/mid`、`fix npt/mid` 在对应语法骨架上新增了热浴类型选择、积分顺序选择和随机项控制

因此，第零章的结论可以概括为：
`fix_nh_middle` 的调用方式并不是脱离原版 `fix nvt`、`fix nph`、`fix npt` 另起一套，而是在原版三类语法框架上分别增加了一组新的控制关键字。
后文第一章说明这些新增关键字背后的实现改动，第二章说明这些关键字在输入文件中应如何具体书写。

## 第一章：程序改动说明

本章汇总 `fix_nh_middle` 相对于原版 `FixNH` 的 10 个关键改动。
每一节均按统一结构组织：

1. 新增功能
2. 关键代码
3. 公式说明

### 改动 1：扩展输入关键字，支持热浴/压浴类型与积分顺序切换

### 新增功能

原版 `FixNH` 并不识别以下关键字：

- `thermostat`
- `barostat`
- `integrator`
- `seed`
- `zero`

`fix_nh_middle` 增加了这些关键字，使得同一套 NPT 框架可以在运行时切换：

- 温度热浴：`nh` 或 `langevin`
- 压力热浴：`nh` 或 `langevin`
- 积分顺序：`side` 或 `middle`
- Langevin 随机种子
- 是否去除质心随机冲量

### 关键代码

首先，需要在调用基类 `FixNH` 之前，从参数列表中移除基类无法识别的新增关键字：

```cpp
FixNHMiddle::ArgList FixNHMiddle::filter_middle_args(int narg, char **arg)
{
  FixNHMiddle::ArgList filtered;
  filtered.storage.reserve(narg);

  auto append = [&filtered](const char *value) { filtered.storage.emplace_back(value); };

  for (int i = 0; i < MIN(narg,3); i++) append(arg[i]);

  int iarg = 3;
  while (iarg < narg) {
    if ((strcmp(arg[iarg],"thermostat") == 0 || strcmp(arg[iarg],"barostat") == 0) &&
        iarg+2 < narg && utils::is_double(arg[iarg+2])) {
      iarg += 3;
    } else if (strcmp(arg[iarg],"thermostat") == 0 || strcmp(arg[iarg],"barostat") == 0 ||
               strcmp(arg[iarg],"seed") == 0 || strcmp(arg[iarg],"zero") == 0) {
      iarg += 2;
    } else if (strcmp(arg[iarg],"integrator") == 0) {
      iarg += 2;
    } else {
      append(arg[iarg]);
      iarg++;
    }
  }

  filtered.argv.reserve(filtered.storage.size());
  for (auto &entry : filtered.storage) filtered.argv.push_back(entry.data());
  return filtered;
}
```

然后，由 `FixNHMiddle` 自身解析这些扩展参数：

```cpp
void FixNHMiddle::parse_middle_args(int narg, char **arg)
{
  nh_temp_flag = 1;
  nh_press_flag = 1;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"thermostat") == 0) {
      if (strcmp(arg[iarg+1],"nh") == 0) nh_temp_flag = 1;
      else if (strcmp(arg[iarg+1],"langevin") == 0) {
        nh_temp_flag = 0;
        langevin_temp_damp_flag = 1;
        damp_t = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        iarg++;
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"barostat") == 0) {
      if (strcmp(arg[iarg+1],"nh") == 0) nh_press_flag = 1;
      else if (strcmp(arg[iarg+1],"langevin") == 0) {
        nh_press_flag = 0;
        langevin_press_damp_flag = 1;
        damp_p = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        iarg++;
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"integrator") == 0) {
      if (strcmp(arg[iarg+1],"side") == 0) integrator = SIDE;
      else if (strcmp(arg[iarg+1],"middle") == 0) integrator = MIDDLE;
      iarg += 2;

    } else if (strcmp(arg[iarg],"seed") == 0) {
      seed = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"zero") == 0) {
      zero_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else {
      iarg++;
    }
  }
}
```

### 公式说明

该改动本身不是单独的动力学方程，而是对演化算符的选择机制进行了扩展。
在理论上，原先固定的 NHC 演化：

$$
\mathcal{L}_T^{\mathrm{NH}}, \qquad \mathcal{L}_B^{\mathrm{NH}}
$$

被扩展为可切换的形式：

$$
\mathcal{L}_T \in \left\{\mathcal{L}_T^{\mathrm{NH}},\ \mathcal{L}_T^{\mathrm{Lan}}\right\}
$$

$$
\mathcal{L}_B \in \left\{\mathcal{L}_B^{\mathrm{NH}},\ \mathcal{L}_B^{\mathrm{Lan}}\right\}
$$

同时积分顺序由单一的 side 格式扩展为：

$$
\text{Integrator} \in \left\{\text{side},\ \text{middle}\right\}
$$

### 改动 2：增加粒子速度的 Langevin 热浴

### 新增功能

`fix_nh_middle` 在原有 Nose-Hoover chain 温度热浴之外，增加了粒子速度的 Langevin O-step。
这一改动使得温度控制不再依赖 NHC 链变量，而可以直接通过 Ornstein-Uhlenbeck 型速度更新实现。

### 关键代码

温度热浴的总入口如下。若 `nh_temp_flag` 为真，则沿用原版 `nhc_temp_integrate()`；否则切换到 `langevin_temp()`：

```cpp
void FixNHMiddle::integrate_temp_thermostat()
{
  if (!tstat_flag) return;
  compute_temp_target();
  update_langevin_coefficients();

  if (nh_temp_flag) nhc_temp_integrate();
  else langevin_temp();
}
```

Langevin 系数在每步开始时根据当前目标温度更新：

```cpp
void FixNHMiddle::update_langevin_coefficients()
{
  double dt = update->dt;
  double dt2 = 0.5 * dt;

  if (tstat_flag && damp_t > 0.0) {
    gamma_t = 1.0 / damp_t;
    lan_c1_t = exp(-gamma_t * dt);
    lan_c2_t = sqrt((1.0 - lan_c1_t * lan_c1_t) * boltz * t_target);
    lan_c1_t_2 = exp(-gamma_t * dt2);
    lan_c2_t_2 = sqrt((1.0 - lan_c1_t_2 * lan_c1_t_2) * boltz * t_target);
  }
}
```

粒子速度的 Langevin 更新为：

```cpp
void FixNHMiddle::langevin_temp()
{
  double lan_coeff1 = (integrator == MIDDLE) ? lan_c1_t : lan_c1_t_2;
  double lan_coeff2 = (integrator == MIDDLE) ? lan_c2_t : lan_c2_t_2;
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double mvv2e = force->mvv2e;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    double mass_i = rmass ? rmass[i] : mass[atom->type[i]];
    double inv_sqrt_m = 1.0 / sqrt(mass_i * mvv2e);
    double kick[3] = {lan_coeff2 * random->gaussian() * inv_sqrt_m,
                      lan_coeff2 * random->gaussian() * inv_sqrt_m,
                      lan_coeff2 * random->gaussian() * inv_sqrt_m};

    v[i][0] = lan_coeff1 * v[i][0] + kick[0];
    v[i][1] = lan_coeff1 * v[i][1] + kick[1];
    v[i][2] = lan_coeff1 * v[i][2] + kick[2];
  }
}
```

### 公式说明

这一实现对应粒子速度的 Langevin 更新公式：

$$
\mathbf{v}_i \leftarrow c_1^{\mathrm{temp}} \mathbf{v}_i
+ \frac{c_2^{\mathrm{temp}}}{\sqrt{m_i}} \boldsymbol{\eta}_i
$$

其中

$$
c_1^{\mathrm{temp}} = \exp(-\gamma_{\mathrm{Lan}}\Delta t)
$$

$$
c_2^{\mathrm{temp}} = \sqrt{k_B T\left(1-(c_1^{\mathrm{temp}})^2\right)}
$$

在代码中：

- `lan_c1_t` 对应 $c_1^{\mathrm{temp}}$
- `lan_c2_t` 对应 $c_2^{\mathrm{temp}}$
- `kick[3]` 对应随机项

由于 LAMMPS 采用内部单位制，程序中显式引入了 `mvv2e`，因此离散实现写成：

$$
\frac{c_2^{\mathrm{temp}}}{\sqrt{m_i \cdot mvv2e}}\boldsymbol{\eta}_i
$$

这与论文中的公式在物理意义上完全等价。

### 改动 3：增加盒子自由度的 Langevin 压浴

### 新增功能

除粒子热浴外，`fix_nh_middle` 还为 barostat 变量增加了 Langevin O-step。
这样盒子自由度同样可以不使用 Nose-Hoover chain，而是直接通过随机阻尼过程控压。

### 关键代码

压力热浴的总入口如下：

```cpp
void FixNHMiddle::integrate_press_thermostat()
{
  if (!pstat_flag || !mpchain) return;

  if (nh_press_flag) nhc_press_integrate();
  else langevin_press();
}
```

压浴系数的更新如下：

```cpp
if (pstat_flag && damp_p > 0.0) {
  gamma_p = 1.0 / damp_p;
  lan_c1_p = exp(-gamma_p * dt);
  lan_c1_p_2 = exp(-gamma_p * dt2);

  double denom = (pstyle == ISO && pdim > 0) ? pdim : 1.0;
  lan_c2_p = sqrt((1.0 - lan_c1_p * lan_c1_p) * boltz * t_target / denom);
  lan_c2_p_2 = sqrt((1.0 - lan_c1_p_2 * lan_c1_p_2) * boltz * t_target / denom);
}
```

盒子自由度的 Langevin 更新如下：

```cpp
void FixNHMiddle::langevin_press()
{
  double lan_coeff1 = (integrator == MIDDLE) ? lan_c1_p : lan_c1_p_2;
  double lan_coeff2 = (integrator == MIDDLE) ? lan_c2_p : lan_c2_p_2;
  double kicks[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  if (comm->me == 0) {
    if (pcouple == XYZ) {
      double kick = lan_coeff2 * random->gaussian() / sqrt(omega_mass[0]);
      kicks[0] = kicks[1] = kicks[2] = kick;
    } else {
      for (int i = 0; i < 6; i++)
        if (p_flag[i]) kicks[i] = lan_coeff2 * random->gaussian() / sqrt(omega_mass[i]);
    }
  }

  MPI_Bcast(kicks, 6, MPI_DOUBLE, 0, world);

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) omega_dot[i] = lan_coeff1 * omega_dot[i] + kicks[i];
}
```

为突出 MPI 同步逻辑，上面的代码块只展开了 `couple xyz` 和非耦合分支；当前源码还分别处理 `couple xy`、`couple yz` 与 `couple xz`，使耦合方向共享同一个随机 kick，未耦合方向独立生成随机 kick。

### 公式说明

这一实现对应 barostat 速度的 Langevin 更新：

$$
v_{\varepsilon} \leftarrow c_1^{\mathrm{press}} v_{\varepsilon}
+ \frac{c_2^{\mathrm{press}}}{\sqrt{W}} \eta
$$

其中

$$
c_1^{\mathrm{press}} = \exp(-\gamma_{\mathrm{Lan}}^V \Delta t)
$$

$$
c_2^{\mathrm{press}} = \sqrt{k_B T\left(1-(c_1^{\mathrm{press}})^2\right)}
$$

在程序中：

- `omega_mass[i]` 承担有效活塞质量 $W$ 的角色
- `omega_dot[i] = lan_coeff1 * omega_dot[i] + kicks[i]` 即离散更新式

因此，代码中

$$
\frac{lan\_coeff2}{\sqrt{\omega\_mass[i]}}
$$

正对应公式中的

$$
\frac{c_2^{\mathrm{press}}}{\sqrt{W}}
$$

### 改动 4：在各向同性压浴中引入 $1/d$ 修正

### 新增功能

`fix_nh_middle` 并未直接照搬标准 MTTK 文献中的 barostat 噪声强度，而是针对 LAMMPS 原版 `FixNH` 的盒子动量分布引入了各向同性情况下的 $1/d$ 修正。
该修正是为了保证当前程序所采用的 box momentum 方程与 Langevin 随机项保持自洽。

### 关键代码

该修正体现在压浴系数更新时的 `denom`：

```cpp
double denom = (pstyle == ISO && pdim > 0) ? pdim : 1.0;
lan_c2_p = sqrt((1.0 - lan_c1_p * lan_c1_p) * boltz * t_target / denom);
lan_c2_p_2 = sqrt((1.0 - lan_c1_p_2 * lan_c1_p_2) * boltz * t_target / denom);
```

当 `pstyle == ISO` 时，`denom = pdim`；在三维各向同性 NPT 中有 `pdim = 3`。

### 公式说明

论文第三章中指出，标准各向同性 MTTK 方程与 LAMMPS 原生 `fix_nh` 实际采用的盒子动量方程并不完全相同，具体参见本文毕业论文第 3.2 节。若用 $p_{\varepsilon}$ 表示盒子动量、$W$ 表示活塞质量、$d$ 表示空间维数，则标准 MTTK 中的盒子动量推进可概括为：

$$
\dot{p}_{\varepsilon}^{\mathrm{MTTK}}
= d\left[
V(\mathcal{P}^{\mathrm{int}}-P)
+ \frac{1}{N_f}\sum_i \frac{\mathbf{p}_i^2}{m_i}
\right]
- \frac{p_{\xi_1}}{Q'_1}p_{\varepsilon}
$$

而 LAMMPS 原生 `fix_nh` 中使用的形式相当于把前面方括号整体的 $d$ 因子去掉：

$$
\dot{p}_{\varepsilon}^{\mathrm{LAMMPS}}
=
V(\mathcal{P}^{\mathrm{int}}-P)
+ \frac{1}{N_f}\sum_i \frac{\mathbf{p}_i^2}{m_i}
- \frac{p_{\xi_1}}{Q'_1}p_{\varepsilon}
$$

与此对应，压浴链第一项的动能反馈也不同。标准 MTTK 使用：

$$
\dot{p}_{\xi_1}^{\mathrm{MTTK}}
=
\frac{p_{\varepsilon}^{2}}{W}
- k_B T
- \frac{p_{\xi_2}}{Q'_2}p_{\xi_1}
$$

而 LAMMPS 原生形式对应为：

$$
\dot{p}_{\xi_1}^{\mathrm{LAMMPS}}
=
\frac{d\,p_{\varepsilon}^{2}}{2W}
- k_B T
- \frac{p_{\xi_2}}{Q'_2}p_{\xi_1}
$$

因此，两套方程对应的盒子动量部分守恒量也不同。标准 MTTK 的盒子动能项为：

$$
\frac{p_{\varepsilon}^{2}}{2W}
$$

而 LAMMPS 原生 `fix_nh` 对应的是：

$$
\frac{d\,p_{\varepsilon}^{2}}{2W}
$$

也就是说，LAMMPS 原生方程给出的盒子动量边缘分布不是

$$
\rho_{\mathrm{MTTK}}(p_{\varepsilon})
\propto
\exp\left[-\beta\frac{p_{\varepsilon}^{2}}{2W}\right]
$$

而是

$$
\rho_{\mathrm{LAMMPS}}(p_{\varepsilon})
\propto
\exp\left[-\beta\frac{d\,p_{\varepsilon}^{2}}{2W}\right]
$$

这意味着在 LAMMPS 风格的盒子动量分布下，$p_{\varepsilon}$ 的平衡方差不是 $W/\beta$，而是 $W/(d\beta)$。因此，如果 Langevin 压浴仍要以 LAMMPS 原生 box momentum 分布为不变分布，就不能直接使用标准 MTTK 的噪声强度，而必须把噪声方差除以 $d$。

对应的 Langevin 压浴 O-step 应写成：

$$
\mathcal{L}_{B}^{\mathrm{Lan}}:
p_{\varepsilon} \leftarrow
\exp(-\gamma_{\mathrm{Lan}}^V \Delta t)\, p_{\varepsilon}
+ \sqrt{\frac{W}{d\beta}\left(1-\exp(-2\gamma_{\mathrm{Lan}}^V \Delta t)\right)}\, \eta
$$

因此噪声系数应写成：

$$
c_2^{\mathrm{press}}
= \sqrt{\frac{1}{d}k_B T\left(1-(c_1^{\mathrm{press}})^2\right)}
$$

代码中的

```cpp
/ denom
```

正是对该 $1/d$ 因子的实现。

### 改动 5：增加压浴随机项的 MPI 全局同步

### 新增功能

粒子速度属于 per-atom 变量，可以在各 MPI rank 本地独立生成随机数。
但 `omega_dot` 是全局盒子自由度，若不同进程使用不同的随机扰动，则后续盒子演化和坐标 remap 将失去一致性。

因此，`fix_nh_middle` 增加了压浴随机项的集中生成和广播机制。

### 关键代码

压浴随机项只在 0 号进程上生成：

```cpp
if (comm->me == 0) {
  if (pcouple == XYZ) {
    double kick = lan_coeff2 * random->gaussian() / sqrt(omega_mass[0]);
    kicks[0] = kicks[1] = kicks[2] = kick;
  } else {
    for (int i = 0; i < 6; i++)
      if (p_flag[i]) kicks[i] = lan_coeff2 * random->gaussian() / sqrt(omega_mass[i]);
  }
}
```

为突出广播机制，上面的片段只展示了 `couple xyz` 和非耦合分支；当前源码还分别处理 `couple xy`、`couple yz` 与 `couple xz`，以保证各耦合方向使用一致的随机扰动。

随后广播给所有进程：

```cpp
MPI_Bcast(kicks, 6, MPI_DOUBLE, 0, world);
```

最后每个进程用同一组随机项更新本地保存的 `omega_dot`：

```cpp
for (int i = 0; i < 6; i++)
  if (p_flag[i]) omega_dot[i] = lan_coeff1 * omega_dot[i] + kicks[i];
```

### 公式说明

这一改动并不改变连续动力学方程本身，而是保证离散更新式在并行实现下满足：

$$
\omega_{\alpha} \leftarrow c_1^{\mathrm{press}}\omega_{\alpha} + \xi_{\alpha}
$$

其中所有 MPI 进程使用同一个 $\xi_{\alpha}$：

$$
\xi_{\alpha}^{(0)} = \xi_{\alpha}^{(1)} = \cdots = \xi_{\alpha}^{(n_{\mathrm{proc}}-1)}
$$

只有满足这一条件，后续盒子变量和粒子 remap 才是全局一致的。

### 改动 6：增加 middle 积分顺序

### 新增功能

原版 `FixNH` 采用 side 型积分顺序。
`fix_nh_middle` 在保留 side 路径的同时，增加了 middle 积分路径，从而可以在同一框架中选择不同的时间分裂方式。

### 关键代码

side 与 middle 的选择由 `integrator` 决定：

```cpp
else if (strcmp(arg[iarg],"integrator") == 0) {
  if (strcmp(arg[iarg+1],"side") == 0) integrator = SIDE;
  else if (strcmp(arg[iarg+1],"middle") == 0) integrator = MIDDLE;
  iarg += 2;
}
```

middle 路径的 `initial_integrate()` 如下：

```cpp
void FixNHMiddle::initial_integrate(int /*vflag*/)
{
  if (integrator == SIDE) {
    initial_integrate_side();
    return;
  }

  if (pstat_flag) nh_v_press();
  nve_v();
  nve_v();
  if (pstat_flag) nh_v_press();

  if (pstat_flag) {
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) {
    compute_press_target();
    update_omega_dot();
  }

  if (pstat_flag) remap();
  nve_x_half();

  integrate_press_thermostat();
  integrate_temp_thermostat();

  nve_x_half();

  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}
```

与之配套的 `final_integrate()` 如下：

```cpp
void FixNHMiddle::final_integrate()
{
  if (integrator == SIDE) {
    final_integrate_side();
    return;
  }

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) update_omega_dot();
}
```

### 公式说明

该实现对应将 NPT 离散推进重新组织为 middle 型算子分裂。
按当前代码的实际顺序，可将其概括为：

$$
\mathcal{L}_{v_r}
\mathcal{L}_{v_t}
\mathcal{L}_{v_t}
\mathcal{L}_{v_r}
\mathcal{L}_{v_{\varepsilon}}
\mathcal{L}_{h,x_r}
\mathcal{L}_{x_t}
\mathcal{L}_{B}
\mathcal{L}_{T}
\mathcal{L}_{x_t}
\mathcal{L}_{h,x_r}
$$

其中：

- $\mathcal{L}_{v_t}$ 由 `nve_v()` 实现
- $\mathcal{L}_{v_r}$ 由 `nh_v_press()` 实现
- $\mathcal{L}_{v_{\varepsilon}}$ 由 `update_omega_dot()` 实现
- $\mathcal{L}_{x_t}$ 由 `nve_x_half()` 实现
- $\mathcal{L}_{h,x_r}$ 由 `remap()` 实现
- $\mathcal{L}_T$ 为粒子热浴
- $\mathcal{L}_B$ 为压浴

### 改动 7：增加 half-step 位置推进

### 新增功能

middle 格式要求位置更新拆分为两个半步。
为此，`fix_nh_middle` 在原版整步位置推进之外，增加了 `nve_x_half()`。

### 关键代码

```cpp
void FixNHMiddle::nve_x_half()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dthalf * v[i][0];
      x[i][1] += dthalf * v[i][1];
      x[i][2] += dthalf * v[i][2];
    }
  }
}
```

### 公式说明

该函数直接实现了位置的半步推进公式：

$$
\mathbf{x}_i \leftarrow \mathbf{x}_i + \frac{\Delta t}{2}\mathbf{v}_i
$$

代码中的三行更新：

```cpp
x[i][0] += dthalf * v[i][0];
x[i][1] += dthalf * v[i][1];
x[i][2] += dthalf * v[i][2];
```

即为该式在三维笛卡尔坐标下的逐分量实现。

### 改动 8：增加新的 barostat 半步推进 `update_omega_dot()`

### 新增功能

为适配当前 middle 路径与 LAMMPS 风格 box momentum 方程，`fix_nh_middle` 没有直接复用原版 `FixNH::nh_omega_dot()`，而是增加了新的 `update_omega_dot()`。
这个函数执行的就是原版 `nh_omega_dot()` 所承担的核心功能：根据当前压力、目标压力、MTK 修正和偏应力项更新 barostat 速度变量 `omega_dot`。改名为 `update_omega_dot()` 是为了让函数名直接表达“更新 `omega_dot`”这一动作，而不是强调某一种特定积分路径。

需要注意的是，`update_omega_dot()` 并不是逐行等价复制原版 `FixNH::nh_omega_dot()`。毕业论文第三章中对 LAMMPS 原生 `fix_nh` 的运动方程有一个重要讨论：LAMMPS 原生 box momentum 方程与标准 MTTK 写法不同，最终对应的盒子动量边缘分布含有空间维数 $d$ 因子，即 $\rho(p_{\varepsilon}) \propto \exp[-\beta d p_{\varepsilon}^{2}/(2W)]$。本文实现默认保留与 LAMMPS 原生方程一致的策略，而不是切换到 Tuckerman 书中的标准 MTTK 方程。

因此，`update_omega_dot()` 在 MTK 项的写法上相对于原版 `nh_omega_dot()` 有细微差别：

- 原版 `FixNH::nh_omega_dot()` 中，各向同性时 `mtk_term1` 由 `tdof * boltz * t_current / (pdim * atom->natoms)` 给出；非各向同性时，对温度 compute 的动能张量分量求和后除以 `pdim * atom->natoms`。
- 当前 `update_omega_dot()` 中，各向同性时直接使用 `boltz * t_current`；非各向同性时，对温度 compute 的动能张量分量求和后除以 `tdof`。
- 原版 `mtk_term2` 使用 `pdim * atom->natoms` 归一化，而当前实现使用 `tdof` 归一化。

这个差别的目的不是改变 `omega_dot` 更新这个职责，而是让 middle 路径中的 MTK 修正与当前采用的自由度定义和 LAMMPS 风格 box momentum 方程保持一致。与第三章的讨论对应，若使用 Langevin 压浴，还需要在压浴随机项中配合 $1/d$ 的噪声强度修正；这一点在前文“改动 4”中由 `lan_c2_p` 的 `/ denom` 实现。

### 关键代码

```cpp
void FixNHMiddle::update_omega_dot()
{
  double volume = (dimension == 3) ? domain->xprd*domain->yprd*domain->zprd : domain->xprd*domain->yprd;
  if (deviatoric_flag) compute_deviatoric();

  mtk_term1 = 0.0;
  if (mtk_flag) {
    if (pstyle == ISO) mtk_term1 = boltz * t_current;
    else {
      double *mvv_current = temperature->vector;
      for (int i = 0; i < 3; i++)
        if (p_flag[i]) mtk_term1 += mvv_current[i];
      mtk_term1 /= tdof;
    }
  }

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      double f_omega = (p_current[i]-p_hydro)*volume / (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega*dthalf;
      omega_dot[i] *= pdrag_factor;
    }

  mtk_term2 = 0.0;
  if (mtk_flag) {
    for (int i = 0; i < 3; i++)
      if (p_flag[i]) mtk_term2 += omega_dot[i];
    if (tdof > 0.0) mtk_term2 /= tdof;
  }
}
```

### 公式说明

这一实现对应 barostat 动量半步推进：

$$
\omega_i \leftarrow \omega_i + f_{\omega_i}\frac{\Delta t}{2}
$$

其中

$$
f_{\omega_i}
=
\frac{(P_i^{\mathrm{int}}-P_i^{\mathrm{target}})V}{W_i}
+ \frac{\mathrm{MTK}}{W_i}
- \frac{f_i^{\mathrm{dev}}}{W_i}
$$

在代码中：

- 压差驱动项由
  ```cpp
  (p_current[i]-p_hydro)*volume / (omega_mass[i] * nktv2p)
  ```
  给出

- MTK 修正项由
  ```cpp
  mtk_term1 / omega_mass[i]
  ```
  给出

- 偏应力项由
  ```cpp
  fdev[i]/(omega_mass[i] * nktv2p)
  ```
  给出

因此，`update_omega_dot()` 是将连续的 box momentum 方程离散为 middle 积分半步推进的核心实现；从职责上看，它对应原版 `FixNH::nh_omega_dot()` 的 `omega_dot` 更新功能。

### 改动 9：增加去除质心随机冲量的功能

### 新增功能

在使用 Langevin 粒子热浴时，随机力会自然激发质心运动。
为了在需要时保持与原版 side Nose-Hoover 的“非质心热化”行为一致，`fix_nh_middle` 增加了 `zero_flag` 控制的质心随机冲量去除步骤。

### 关键代码

在产生随机 kick 后，先累积质量加权总冲量：

```cpp
double fsum[4] = {0.0, 0.0, 0.0, 0.0};
double fsumall[4] = {0.0, 0.0, 0.0, 0.0};

for (int i = 0; i < nlocal; i++) {
  double mass_i = rmass ? rmass[i] : mass[atom->type[i]];
  double inv_sqrt_m = 1.0 / sqrt(mass_i * mvv2e);
  double kick[3] = {lan_coeff2 * random->gaussian() * inv_sqrt_m,
                    lan_coeff2 * random->gaussian() * inv_sqrt_m,
                    lan_coeff2 * random->gaussian() * inv_sqrt_m};

  if (zero_flag) {
    fsum[0] += mass_i * kick[0];
    fsum[1] += mass_i * kick[1];
    fsum[2] += mass_i * kick[2];
    fsum[3] += mass_i;
  }

  v[i][0] = lan_coeff1 * v[i][0] + kick[0];
  v[i][1] = lan_coeff1 * v[i][1] + kick[1];
  v[i][2] = lan_coeff1 * v[i][2] + kick[2];
}
```

随后在全局范围内汇总总冲量，并统一扣除：

```cpp
if (zero_flag) {
  MPI_Allreduce(fsum, fsumall, 4, MPI_DOUBLE, MPI_SUM, world);
  if (fsumall[3] > 0.0) {
    double correction[3] = {fsumall[0]/fsumall[3], fsumall[1]/fsumall[3], fsumall[2]/fsumall[3]};
    for (int i = 0; i < nlocal; i++) {
      v[i][0] -= correction[0];
      v[i][1] -= correction[1];
      v[i][2] -= correction[2];
    }
  }
}
```

### 公式说明

该步骤对应如下修正过程。
首先定义每个原子的随机速度增量：

$$
\delta \mathbf{v}_{\mathrm{ran}}[i]
= \frac{c_2^{\mathrm{temp}}}{\sqrt{m_i}}\boldsymbol{\eta}_i
$$

则全体系随机总冲量为：

$$
\mathbf{I}_{\mathrm{tot}} = \sum_{i=1}^{N} m_i \delta \mathbf{v}_{\mathrm{ran}}[i]
$$

总质量为：

$$
M_{\mathrm{tot}} = \sum_{i=1}^{N} m_i
$$

修正后的速度更新为：

$$
\mathbf{v}_i \leftarrow \mathbf{v}_i - \frac{\mathbf{I}_{\mathrm{tot}}}{M_{\mathrm{tot}}}
$$

因此满足：

$$
\sum_i m_i \Delta \mathbf{v}_i = 0
$$

即随机项不再向质心注入净动量。

### 改动 10：增加 `zero_flag` 与温度自由度的自动联动

### 新增功能

在使用粒子 Langevin 热浴时，`zero_flag` 不仅控制是否去除质心随机冲量，还同时控制内部温度 compute 的 `extra/dof` 设置。
其行为可以概括为：

- `zero yes` 时，保留当前温度 compute 的默认 `extra/dof` 设置
- `zero no` 时，自动将内部温度 compute 的 `extra/dof` 改为 0
- 当温度热浴仍为 Nose-Hoover chain 时，`apply_zero_dof_mode()` 会直接返回，`zero` 不改变温度 compute 的自由度设置

### 关键代码

```cpp
void FixNHMiddle::apply_zero_dof_mode()
{
  if (!temperature || nh_temp_flag) return;

  if (zero_flag) {
    temperature->reset_extra_dof();
    temperature->setup();
    return;
  }

  char *args[2];
  args[0] = const_cast<char *>("extra/dof");
  args[1] = const_cast<char *>("0");
  temperature->modify_params(2, args);
  temperature->setup();
}
```

该函数在初始化和 `setup()` 时自动调用：

```cpp
void FixNHMiddle::init()
{
  FixNH::init();
  apply_zero_dof_mode();
  update_langevin_coefficients();
}

void FixNHMiddle::setup(int vflag)
{
  FixNH::setup(vflag);
  apply_zero_dof_mode();
  update_langevin_coefficients();
}
```

### 公式说明

这一改动对应温度定义中自由度参数的调整。
温度与动能之间的一般关系仍写作：

$$
K = \frac{N_f}{2}k_B T
$$

但在实际模拟中，$N_f$ 未必只是简单的平动自由度计数，因为体系还可能叠加其他约束或额外自由度修正。
因此，`fix_nh_middle` 在这里并不直接假定某个固定的 $N_f$，而是通过修改内部温度 compute 的 `extra/dof` 参数来控制温度定义所使用的自由度数。

就实现而言，在 `thermostat langevin` 路径下，`apply_zero_dof_mode()` 的作用仅需理解为：

- `zero yes`：保留默认的 `extra/dof`
- `zero no`：自动执行与 `compute_modify ... extra/dof 0` 等价的处理

## 第二章：输入文件的写法

本章说明如何在 LAMMPS 输入脚本中调用 `fix nvt/mid`、`fix nph/mid` 与 `fix npt/mid`。
由于三者都继承自 `FixNHMiddle`，而 `FixNHMiddle` 又继承自 `FixNH`，因此其输入格式分为两部分：

1. 原版 `fix nvt` / `fix nph` / `fix npt` 已有的温度和压力控制关键字
2. `fix_nh_middle` 新增的扩展关键字

### 2.1 基本规则

三个 `mid` 版本的基本规则如下：

- `fix nvt/mid` 必须包含 `temp`，且不能包含压力控制关键字
- `fix nph/mid` 必须包含压力控制关键字，且不能包含 `temp`
- `fix npt/mid` 必须同时包含 `temp` 和压力控制关键字

这里的压力控制关键字包括：

- `iso`
- `aniso`
- `tri`
- `x`
- `y`
- `z`
- `xy`
- `xz`
- `yz`

这是因为三者分别模仿了原版 `FixNVT`、`FixNPH`、`FixNPT` 的外层封装逻辑。

`FixNVTMid`：

```cpp
FixNVTMid::FixNVTMid(LAMMPS *lmp, int narg, char **arg) : FixNHMiddle(lmp, narg, arg)
{
  if (!tstat_flag) error->all(FLERR, "Temperature control must be used with fix nvt/mid");
  if (pstat_flag) error->all(FLERR, "Pressure control can not be used with fix nvt/mid");

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp", id_temp, group->names[igroup]));
  tcomputeflag = 1;
}
```

`FixNPHMid`：

```cpp
FixNPHMid::FixNPHMid(LAMMPS *lmp, int narg, char **arg) : FixNHMiddle(lmp, narg, arg)
{
  if (tstat_flag) error->all(FLERR, "Temperature control can not be used with fix nph/mid");
  if (!pstat_flag) error->all(FLERR, "Pressure control must be used with fix nph/mid");

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp", id_temp));
  tcomputeflag = 1;

  id_press = utils::strdup(std::string(id) + "_press");
  modify->add_compute(fmt::format("{} all pressure {}", id_press, id_temp));
  pcomputeflag = 1;
}
```

`FixNPTMid`：

```cpp
FixNPTMid::FixNPTMid(LAMMPS *lmp, int narg, char **arg) : FixNHMiddle(lmp, narg, arg)
{
  if (!tstat_flag) error->all(FLERR, "Temperature control must be used with fix npt/mid");
  if (!pstat_flag) error->all(FLERR, "Pressure control must be used with fix npt/mid");

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp", id_temp));
  tcomputeflag = 1;

  id_press = utils::strdup(std::string(id) + "_press");
  modify->add_compute(fmt::format("{} all pressure {}", id_press, id_temp));
  pcomputeflag = 1;
}
```

这表明：

- `fix nvt/mid` 会自动创建温度 compute
- `fix nph/mid` 会自动创建温度和压强 compute
- `fix npt/mid` 会自动创建温度和压强 compute
- 三者的用法均应与对应的原版 fix 保持一致，只是在此基础上增加扩展关键字

### 2.2 通用语法

三个入口的通用写法分别如下。

`fix nvt/mid`：

```lammps
fix ID group-ID nvt/mid temp Tstart Tstop Tdamp \
                   [thermostat nh|langevin [Tdamp_lan]] \
                   [integrator side|middle] \
                   [seed integer] \
                   [zero yes/no]
```

`fix nph/mid`：

```lammps
fix ID group-ID nph/mid PRESS_KEYWORD Pstart Pstop Pdamp \
                   [barostat nh|langevin [Pdamp_lan]] \
                   [integrator side|middle] \
                   [seed integer]
```

`fix npt/mid`：

```lammps
fix ID group-ID npt/mid temp Tstart Tstop Tdamp \
                   PRESS_KEYWORD Pstart Pstop Pdamp \
                   [thermostat nh|langevin [Tdamp_lan]] \
                   [barostat nh|langevin [Pdamp_lan]] \
                   [integrator side|middle] \
                   [seed integer] \
                   [zero yes/no]
```

其中：

- `temp Tstart Tstop Tdamp` 是原版 `fix nvt` / `fix npt` 的温度参数
- `PRESS_KEYWORD` 可以是：
  - `iso`
  - `aniso`
  - `tri`
  - `x`
  - `y`
  - `z`
  - `xy`
  - `xz`
  - `yz`
- `thermostat` 和 `barostat` 是新增关键字
- `integrator` 是新增关键字
- `seed` 是 Langevin 随机种子
- `zero` 控制是否去除质心随机冲量

### 2.3 新增关键字的精确含义

关键字解析逻辑如下：

```cpp
if (strcmp(arg[iarg],"thermostat") == 0) {
  if (strcmp(arg[iarg+1],"nh") == 0) nh_temp_flag = 1;
  else if (strcmp(arg[iarg+1],"langevin") == 0) {
    nh_temp_flag = 0;
    langevin_temp_damp_flag = 1;
    damp_t = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    iarg++;
  }
  iarg += 2;

} else if (strcmp(arg[iarg],"barostat") == 0) {
  if (strcmp(arg[iarg+1],"nh") == 0) nh_press_flag = 1;
  else if (strcmp(arg[iarg+1],"langevin") == 0) {
    nh_press_flag = 0;
    langevin_press_damp_flag = 1;
    damp_p = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    iarg++;
  }
  iarg += 2;

} else if (strcmp(arg[iarg],"integrator") == 0) {
  if (strcmp(arg[iarg+1],"side") == 0) integrator = SIDE;
  else if (strcmp(arg[iarg+1],"middle") == 0) integrator = MIDDLE;
  iarg += 2;

} else if (strcmp(arg[iarg],"seed") == 0) {
  seed = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
  iarg += 2;

} else if (strcmp(arg[iarg],"zero") == 0) {
  zero_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
  iarg += 2;
}
```

因此应当注意以下几点。

#### `thermostat`

可选值：

- `thermostat nh`
- `thermostat langevin Tdamp_lan`

含义：

- 若取 `nh`，粒子热浴使用原版 Nose-Hoover chain
- 若取 `langevin`，粒子热浴使用 `langevin_temp()`

注意：

- 当写成 `thermostat langevin` 时，后面必须紧跟一个数值参数，表示 Langevin 温度热浴弛豫时间

#### `barostat`

可选值：

- `barostat nh`
- `barostat langevin Pdamp_lan`

含义：

- 若取 `nh`，盒子热浴使用原版 barostat NHC
- 若取 `langevin`，盒子自由度使用 `langevin_press()`

注意：

- 当写成 `barostat langevin` 时，后面必须紧跟一个数值参数，表示 Langevin 压浴弛豫时间
- `barostat` 关键字只对包含压力控制的 `fix nph/mid` 和 `fix npt/mid` 有意义

#### `integrator`

可选值：

- `integrator side`
- `integrator middle`

含义：

- `side` 对应 side 型时间分裂
- `middle` 对应 `fix_nh_middle` 的核心 middle 分裂

若未显式指定，则默认值为 `middle`，因为构造函数中初始化为：

```cpp
integrator(MIDDLE)
```

#### `seed`

写法：

```lammps
seed 123456
```

含义：

- 指定 `RanMars` 的随机数种子
- 仅在使用 Langevin 热浴或压浴时才真正影响轨迹

#### `zero`

写法：

```lammps
zero yes
```

或

```lammps
zero no
```

源码也接受逻辑值 `1/0`，因为底层调用的是：

```cpp
zero_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
```

含义：

- `zero yes`：去除粒子随机项产生的质心净冲量
- `zero no`：保留质心热化

在 `thermostat langevin` 路径下，`zero no` 还会自动把内部温度 compute 的 `extra/dof` 改为 0。
这等价于自动执行：

```lammps
compute_modify myTemp extra/dof 0
```

其中 `myTemp` 在当前实现里并不是用户手写的 compute ID，而是 fix 自动创建的内部温度 compute。

此外，`zero` 关键字只对存在粒子 Langevin 热浴的情形有意义，因此它主要用于：

- `fix nvt/mid`
- `fix npt/mid`

对于 `fix nph/mid`，由于没有温度热浴，通常不需要讨论 `zero`。

### 2.4 最小可用输入示例

最简单的 NVT 写法如下：

```lammps
fix 1 all nvt/mid temp 300.0 300.0 200.0
```

最简单的 NPH 写法如下：

```lammps
fix 1 all nph/mid iso 1.0 1.0 1000.0
```

最简单的各向同性 NPT 写法如下：

```lammps
fix 1 all npt/mid temp 300.0 300.0 200.0 iso 1.0 1.0 1000.0
```

三者分别表示：

- `nvt/mid`：只控温
- `nph/mid`：只控压
- `npt/mid`：同时控温控压

对扩展关键字而言，默认值为：

- `thermostat nh`
- `barostat nh`
- `integrator middle`
- `zero yes`

### 2.5 使用 middle + 双 Langevin 的输入示例

对于 `fix nvt/mid`，若希望使用 middle + Langevin 温度热浴，可写为：

```lammps
fix 1 all nvt/mid temp 300.0 300.0 200.0 \
               integrator middle \
               thermostat langevin 200.0 \
               seed 123456 \
               zero yes
```

对于 `fix nph/mid`，若希望使用 middle + Langevin 压浴，可写为：

```lammps
fix 1 all nph/mid iso 1.0 1.0 1000.0 \
               integrator middle \
               barostat langevin 1000.0 \
               seed 123456
```

若希望显式使用 middle 格式，并将粒子热浴与压浴都设置为 Langevin，可写为：

```lammps
fix 1 all npt/mid temp 300.0 300.0 200.0 \
               iso 1.0 1.0 1000.0 \
               integrator middle \
               thermostat langevin 200.0 \
               barostat langevin 1000.0 \
               seed 123456 \
               zero yes
```

其中：

- `temp ... 200.0` 是原版 NPT 的温度阻尼参数
- `thermostat langevin 200.0` 是 Langevin 温度热浴的弛豫时间
- `iso ... 1000.0` 是原版 NPT 的压强阻尼参数
- `barostat langevin 1000.0` 是 Langevin 压浴的弛豫时间

这两组时间参数在当前实现中是分别解析的，不应混淆。

### 2.6 使用 side + NH 的输入示例

`fix nvt/mid` 的 side + NH 写法：

```lammps
fix 1 all nvt/mid temp 300.0 300.0 200.0 \
               integrator side \
               thermostat nh
```

`fix nph/mid` 的 side + NH 写法：

```lammps
fix 1 all nph/mid iso 1.0 1.0 1000.0 \
               integrator side \
               barostat nh
```

`fix npt/mid` 的 side + NH 写法：

```lammps
fix 1 all npt/mid temp 300.0 300.0 200.0 \
               iso 1.0 1.0 1000.0 \
               integrator side \
               thermostat nh \
               barostat nh
```

这条命令说明：

- 类仍然是 `FixNHMiddle`
- 但热浴、压浴和时间分裂都退回到更接近原版的使用方式

### 2.7 各向异性与三斜盒子的写法

由于底层仍继承自 `FixNH`，因此压力控制关键字的写法与原版 `fix npt` 保持一致。

#### 各向异性正交盒子

```lammps
fix 1 all npt/mid temp 300.0 300.0 200.0 \
               aniso 1.0 1.0 1000.0 \
               integrator middle \
               thermostat langevin 200.0 \
               barostat langevin 1000.0
```

#### 三斜盒子

```lammps
fix 1 all npt/mid temp 300.0 300.0 200.0 \
               tri 1.0 1.0 1000.0 \
               integrator middle \
               thermostat langevin 200.0 \
               barostat langevin 1000.0
```

#### 分量逐项指定

```lammps
fix 1 all npt/mid temp 300.0 300.0 200.0 \
               x 1.0 1.0 1000.0 \
               y 1.0 1.0 1000.0 \
               z 1.0 1.0 1000.0 \
               couple none \
               integrator middle
```

这类写法与原版 `fix npt` 的语法一致，因为这些关键字仍由基类 `FixNH` 解析。

### 2.8 是否需要额外设置温度自由度

通常不需要手动再写：

```lammps
compute_modify myTemp extra/dof 0
```

原因是 `thermostat langevin` 路径下，`fix_nh_middle` 已经通过 `apply_zero_dof_mode()` 自动处理了这一点：

```cpp
if (zero_flag) {
  temperature->reset_extra_dof();
  temperature->setup();
  return;
}

char *args[2];
args[0] = const_cast<char *>("extra/dof");
args[1] = const_cast<char *>("0");
temperature->modify_params(2, args);
temperature->setup();
```

因此：

- `zero yes` 时，保留当前温度 compute 的默认 `extra/dof` 设置
- `zero no` 时，自动执行与 `compute_modify ... extra/dof 0` 等价的处理
- 该自动处理不适用于默认的 `thermostat nh` 路径

### 2.9 推荐的输入模板

若目标是使用本文实现的核心功能，即 middle + Langevin + 去质心随机冲量，推荐模板如下：

```lammps
fix 1 all npt/mid temp Tstart Tstop Tdamp \
               iso Pstart Pstop Pdamp \
               integrator middle \
               thermostat langevin Tdamp_lan \
               barostat langevin Pdamp_lan \
               seed 123456 \
               zero yes
```

若希望在粒子 Langevin 热浴下保留质心热化，并同时将内部温度 compute 的 `extra/dof` 设为 0，则可写为：

```lammps
fix 1 all npt/mid temp Tstart Tstop Tdamp \
               iso Pstart Pstop Pdamp \
               integrator middle \
               thermostat langevin Tdamp_lan \
               barostat langevin Pdamp_lan \
               seed 123456 \
               zero no
```

这两种写法的唯一区别是：

- `zero yes`：去除随机项带来的质心净冲量
- `zero no`：不去除该质心净冲量，并在 `thermostat langevin` 路径下自动执行与 `compute_modify ... extra/dof 0` 等价的处理

## 第三章：完整简化测试样例

本章给出两个可以直接阅读和改写的完整输入样例。它们来自 `/home/shuaixiao/NPT/test/mid` 中的测试模板，但把 `${...}` 形式的待定参数都填成具体数值，并删去了批量测试脚本需要、但理解 `fix_nh_middle` 时不必要的部分。

使用这些样例时需要注意三点：

- 数据文件需要放在运行目录中，例如 `data.argon` 或 `data.water`
- 样例中的 `run` 步数被缩短，目的是快速检查输入文件是否清楚，而不是给出生产模拟长度
- `zero` 统一使用 `yes/no` 写法；旧测试脚本中出现的数字写法只是历史写法

### 3.1 Argon：`npt/mid` + Langevin 热浴和压浴

这个样例由 `test/mid/argon.lmp` 和 `test/mid/argon_mid_iso_114516/argon_fix.in` 简化得到。它对应的核心 fix 命令是：

```lammps
fix 1 all npt/mid thermostat langevin 432.9 barostat langevin 4329.0 zero yes temp 299.5 299.5 432.9 iso 705.423 705.423 4329.0 integrator middle
```

完整输入文件可以写成：

```lammps
# Argon: middle NPT with Langevin thermostat and Langevin barostat

units real
dimension 3
boundary p p p
atom_style full
newton on

pair_style lj/cut 10.0
pair_modify tail yes
read_data data.argon

pair_coeff 1 1 0.238067 3.405
mass 1 39.962

neighbor 1.0 bin
neigh_modify delay 0 every 1 check yes one 4000 page 200000

timestep 2.156

compute mytemp all temp
compute myvirial all pressure mytemp virial
variable scalar_virial equal c_myvirial[1]+c_myvirial[2]+c_myvirial[3]

fix 1 all npt/mid thermostat langevin 432.9 barostat langevin 4329.0 zero yes temp 299.5 299.5 432.9 iso 705.423 705.423 4329.0 integrator middle
fix_modify 1 temp mytemp

thermo 1000
thermo_style custom step temp press pe ke etotal v_scalar_virial vol density pxx pyy pzz
thermo_modify temp mytemp flush yes

dump 1 all custom 50000 argon_mid.lammpstrj id type x y z vx vy vz fx fy fz
dump_modify 1 sort id

run 200000
write_data argon_mid_final.data
```

这条输入中最值得关注的是 `fix 1 all npt/mid ...` 这一行：

- `thermostat langevin 432.9` 表示粒子速度热浴使用 Langevin 形式，阻尼时间为 $432.9$ fs
- `barostat langevin 4329.0` 表示盒子动量变量的压浴使用 Langevin 形式，阻尼时间为 $4329.0$ fs
- `zero yes` 表示每一步去除 Langevin 随机力带来的总质心冲量
- `temp 299.5 299.5 432.9` 表示目标温度从 $299.5$ K 到 $299.5$ K，NH 温度阻尼参数保留为 $432.9$ fs
- `iso 705.423 705.423 4329.0` 表示各向同性控压，三个方向使用同一个体积自由度，目标压力为 $705.423$ atm
- `integrator middle` 表示使用本文新增的 middle 分裂顺序

如果要检查各向异性压力控制，只需要把 `iso 705.423 705.423 4329.0` 改成 `aniso 705.423 705.423 4329.0`；如果要检查三斜盒子的六个盒子自由度，则改成 `tri 705.423 705.423 4329.0`，并在 fix 前加入 `change_box all triclinic`。

### 3.2 q-SPC/Fw 水：`npt/mid` + 柔性水模型

这个样例由 `test/mid/q-SPC_fw.lmp` 和 `run_folder_sim.py` 中 `mid_iso` 的 fix 生成逻辑简化得到。`q-SPC_fw.lmp` 是柔性水模型，因此这里保留了 `bond_style harmonic` 和 `angle_style harmonic`，不使用 SHAKE 约束。

完整输入文件可以写成：

```lammps
# q-SPC/Fw water: middle NPT with Langevin thermostat and Langevin barostat

units real
dimension 3
boundary p p p
atom_style full
newton on

pair_style lj/cut/coul/long 8.5
kspace_style pppm 1.0e-4
bond_style harmonic
angle_style harmonic
pair_modify tail yes

read_data data.water

mass 1 15.9994
mass 2 1.008

pair_coeff 1 1 0.1554252 3.165492
pair_coeff 1 2 0.0 0.0
pair_coeff 2 2 0.0 0.0

bond_coeff 1 1059.162 1.0
angle_coeff 1 75.90 112.0

neighbor 1.0 bin
neigh_modify delay 0 every 1 check yes

timestep 1.0

compute mytemp all temp
compute myvirial all pressure mytemp virial
variable scalar_virial equal c_myvirial[1]+c_myvirial[2]+c_myvirial[3]

fix 1 all npt/mid thermostat langevin 200.0 barostat langevin 200.0 zero yes temp 298.15 298.15 200.0 iso 0.986923 0.986923 200.0 integrator middle
fix_modify 1 temp mytemp

thermo 200
thermo_style custom step temp press pe ke etotal v_scalar_virial vol density pxx pyy pzz
thermo_modify temp mytemp flush yes

dump 1 all custom 5000 water_mid.lammpstrj id mol type q x y z vx vy vz fx fy fz
dump_modify 1 sort id

run 100000
write_data water_mid_final.data
```

这条输入中最重要的参数含义是：

- `temp 298.15 298.15 200.0` 表示目标温度固定在 $298.15$ K，温度阻尼时间为 $200.0$ fs
- `iso 0.986923 0.986923 200.0` 表示各向同性控压，压力单位是 `real` 单位制下的 atm，因此 $0.986923$ atm 约等于 $1$ bar
- `thermostat langevin 200.0` 和 `barostat langevin 200.0` 表示粒子热浴与盒子压浴都使用 Langevin 形式
- `zero yes` 表示去除粒子 Langevin 随机力的总质心冲量
- `integrator middle` 表示使用本文新增的 middle 积分顺序

如果只想测试 NPH middle 压浴，可以删去 `thermostat langevin 200.0 zero yes temp 298.15 298.15 200.0`，并把 fix 类型改为 `nph/mid`：

```lammps
fix 1 all nph/mid barostat langevin 200.0 iso 0.986923 0.986923 200.0 integrator middle
```

这个 NPH 写法对应 `test/mid/water_mid_nph_iso_63001/water_fix.in`，适合单独检查 `update_omega_dot()`、`nhc_press_integrate()` 和盒子动量变量相关的更新。

# 第四章：LAMMPS RIGID 包中的位置约束（代码阅读参考）

> 本章整理 LAMMPS `RIGID` 包中三种"位置约束"机制的代码实现，供阅读
> `fix_nh_middle` 时对照参考。`fix_nh_middle` 的 middle 方案把约束拆成
> 位置约束（C1）与速度约束（C2）两个独立时刻处理；RIGID 包中的
> `fix_shake`、`fix_rattle`、`fix_rigid` 恰好覆盖了从"仅位置约束"到
> "刚体约束"的三种层次。

## 4.1 `fix_shake` 的坐标约束（middle 模式，仅位置）

`fix_shake` 是 RIGID 包中最基础的约束 fix。本仓库的 `fix_shake` 增加了
`middle yes` 开关（对应 `middle_constraints = 1`），使 SHAKE 只做
**坐标（位置）约束**，不再在步末投影速度——这正是 middle 方案中
C1（SHAKE 位置约束）的 LAMMPS 形态。

### 生命周期中的三个挂载点

```cpp
// fix_shake.cpp
void FixShake::initial_integrate(int vflag)   // 积分器移动原子之前
{
  if (!middle_constraints) return;
  for (int i = 0; i < nlocal; i++) {
    if (shake_flag[i]) {
      xref[i][0] = x[i][0]; ...              // 保存参考坐标 xref
    }
  }
}

void FixShake::post_integrate()              // 积分器移动原子之后、算力之前
{
  if (!middle_constraints) return;
  pre_neighbor();
  correct_coordinates_middle(0);             // 把坐标投影回约束几何
}

void FixShake::end_of_step()                 // 步末：middle 模式不做任何事
{
  if (!middle_constraints) return;
  // correct_velocities();  // 现在不希望在步末应用速度约束，
  //                        // 只做坐标约束
}
```

- `initial_integrate()`：在位置半步之前保存 `xref`，作为约束迭代的参考态。
- `post_integrate()`：位置推进之后立刻把原子坐标"投影"回约束几何——
  这是 middle 方案 C1 的位置。
- `end_of_step()`：明确为空。源码注释说明该实现**故意不做**步末速度
  投影（即不需要 RATTLE 类的 C2），与 `fix_nh_middle` 的
  "SHAKE-only" 中间方案一致。

### `correct_coordinates_middle()` 的实现要点

```cpp
void FixShake::correct_coordinates_middle(int vflag)
{
  ...
  // 1) 暂存当前力，把 f 清零；把当前坐标存入 xshake
  for (int i = 0; i < nlocal; i++) { ftmp[i][k] = f[i][k]; f[i][k] = 0.0; ... }

  // 2) 交换 x 与 xref：SHAKE 迭代基于“参考坐标”工作
  double **x_current = x;
  x = xref;
  ...
  // 3) 对每个 SHAKE 簇计算约束力（与普通 SHAKE 相同）
  for (int i = 0; i < nlist; i++) {
    m = list[i];
    if (shake_flag[m] == 2) shake(i);        // 2 原子
    else if (shake_flag[m] == 3) shake3(i);  // 3 原子
    else if (shake_flag[m] == 4) shake4(i);  // 4 原子
    else shake3angle(i);                     // 3 原子 + 1 角
  }

  // 4) 用约束力修正当前（未约束）坐标：x += (dt²/m) * f
  double dtfsq = (rattle ? 0.5 : 1.0) * update->dt * update->dt * force->ftm2v;
  for (int i = 0; i < nlocal; i++) {
    x_current[i][k] += dtfsq / mass[type[i]] * f[i][k];   // 或 rmass
  }

  // 5) 恢复力、恢复 x，把 xshake 指向修正后的坐标
  ...
}
```

要点：

- 约束迭代在**参考坐标** `xref` 上进行（等价于把"无约束位移"投影回约束面）；
- 位置修正量为 `Δx = dt²/m · f_constr`（`dtfsq = dt²·ftm2v`，RATTLE 兼容
  模式为 `0.5·dt²`）；
- 这一修正即 SHAKE 的位置约束力回代，与 GROMACS 中 SHAKE（C1）的
  `cshake()` 位置校正等价。

## 4.2 `fix_rattle` 的位置约束

`fix_rattle` 是 RATTLE 算法：第一遍做 SHAKE（位置约束），第二遍做
RATTLE（速度约束）。RIGID 包中它与 `fix_shake` 共享簇列表与
`shake_flag` 约定。

```cpp
// fix_rattle.cpp
// 初始化：先建立 SHAKE 约束（位置约束部分复用 fix_shake 的簇逻辑）
//   ... "initialize SHAKE first" ...

// 速度约束（C2 等价）发生在 initial_integrate：
//   先做半步无约束速度更新，再对每个簇投影速度
void FixRattle::initial_integrate(int vflag)
{
  if (middle_constraints) comm_mode = XSHAKE;
  ...
  unconstrained_update();           // 半步无约束速度更新
  comm->forward_comm(this);
  for (int m : list) {
    if (shake_flag[m] == 2) vrattle2(m);   // 2 原子速度投影
    else if (shake_flag[m] == 3) vrattle3(m);
    else if (shake_flag[m] == 4) vrattle4(m);
  }
}
```

- **位置约束**：由 SHAKE 部分完成（与 4.1 相同）。
- **速度约束**：`vrattle2/3/4` 把速度投影到键方向，使
  `(v_i − v_j)·r_ij = 0`——对应 middle 方案中完整的 RATTLE（C1+C2）。
- `middle_constraints` 分支下 `comm_mode = XSHAKE`，与 `fix_shake`
  的 middle 模式保持同一通信约定。

## 4.3 `fix_rigid` 的刚体位置约束

`fix_rigid` 把一组原子绑定为一个刚体。其"位置约束"不是迭代校正，
而是**解析投影**：刚体的平动（质心 `xcm`）与转动（四元数 `quat`）
由体运动方程推进，所有原子坐标始终由
`x_i = Xcm + R(q)·r_rel,i` 从体坐标系投影出来。

```cpp
// fix_rigid.cpp
void FixRigid::initial_integrate(int vflag)
{
  for (int ibody = 0; ibody < nbody; ibody++) {
    // 平动：vcm 半步、xcm 全步（速度 Verlet 风格）
    vcm[ibody] += dtf / masstotal[ibody] * fcm[ibody];
    xcm[ibody] += dtv * vcm[ibody];

    // 转动：angmom 半步，然后四元数全步（Richardson 迭代）
    angmom[ibody] += dtf * torque[ibody];
    MathExtra::angmom_to_omega(angmom, ex, ey, ez, inertia, omega);
    MathExtra::richardson(quat[ibody], angmom[ibody], omega[ibody],
                          inertia[ibody], dtq);   // 归一化四元数
    MathExtra::q_to_exyz(quat[ibody], ex, ey, ez); // 更新体轴
  }
  ...
  set_xv();   // 把原子坐标/速度按新四元数投影回体坐标系 —— 位置约束落点
}

void FixRigid::pre_neighbor()
{
  for (int ibody = 0; ibody < nbody; ibody++)
    domain->remap(xcm[ibody], imagebody[ibody]);  // 质心 remap 回周期盒
  image_shift();
}
```

要点：

- 四元数用 `MathExtra::richardson()` 做全步迭代积分（同时给出半步的
  角速度 `omega`），随后 `q_to_exyz()` 更新体轴向量；
- `set_xv()` 是刚体"位置约束"的落点：所有体原子坐标由四元数解析确定，
  不存在迭代；
- `pre_neighbor()` 把 `xcm` remap 回周期盒并同步镜像标志，避免刚体
  整体漂出盒子导致数值问题。

## 4.4 三种位置约束机制与 middle 方案的对照

| 机制 | 位置约束 | 速度约束 | 与 `fix_nh_middle` / GROMACS 对应 |
|---|---|---|---|
| `fix_shake` + `middle yes` | `correct_coordinates_middle()`（`post_integrate`） | 无（`end_of_step` 跳过） | C1：SHAKE 位置约束（GROMACS `cshake`） |
| `fix_rattle` | SHAKE 位置半步 | `vrattle2/3/4` 速度投影 | C1 + C2：完整 RATTLE |
| `fix_rigid` | `set_xv()` 刚体解析投影 | 刚体角速度（体自由度） | 刚体约束（无迭代） |

- 若只关心 middle 方案的**位置约束**（C1），对应实现是 `fix_shake`
  的 `middle yes` 模式：在位置推进之后、算力之前投影坐标，步末不做
  速度约束。
- 若需要完整的键长约束（位置+速度），对应 `fix_rattle`。
- 若约束对象是整体刚体（水分子整体固定），对应 `fix_rigid` 的解析投影。

本章代码均出自本仓库 `src/RIGID/`（`fix_shake.cpp`、`fix_rattle.cpp`、
`fix_rigid.cpp`），阅读时可配合 `fix_nh_middle.cpp` 中约束相关调用点
一起对照。
