# `fix_nh_middle` Manual

本文档面向代码读者，说明 `fix_nh_middle` 相对于原版 `FixNH` 的关键改动，并给出 `fix nvt/mid`、`fix nph/mid`、`fix npt/mid` 的输入文件写法。  
这三个用户入口分别对应：

- `fix nvt/mid` -> `FixNVTMid`
- `fix nph/mid` -> `FixNPHMid`
- `fix npt/mid` -> `FixNPTMid`

三者的核心实现均位于 `FixNHMiddle`。

全文分为三章：

0. 第零章说明原版 `FixNH` 与原版 `fix npt` 的基本功能和调用方式
1. 第一章说明程序实现中的 10 个关键改动
2. 第二章说明输入文件的写法与调用规则

若读者当前只关心如何在 LAMMPS 输入脚本中调用 `fix nvt/mid`、`fix nph/mid` 或 `fix npt/mid`，可直接跳转到第二章。

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

因此，原版 `FixNH` 原生支持的主要关键字包括：

- `temp`
- `iso`
- `aniso`
- `tri`
- `x`
- `y`
- `z`
- `xy`
- `xz`
- `yz`
- `couple`
- `drag`
- 以及若干更细的高级选项，例如 `tchain`、`pchain`、`mtk`、`ploop`、`tloop`、`nreset`、`dilate`、`ptemp`

而后文 `fix_nh_middle` 新增的：

- `thermostat`
- `barostat`
- `integrator`
- `seed`
- `zero`

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

原版 `fix nph` 的最常见写法为：

```lammps
fix 1 all nph iso 1.0 1.0 1000.0
```

这条命令只包含压力控制，不包含 `temp` 关键字。

原版 `fix npt` 的最常见调用写法为：

```lammps
fix 1 all npt temp 300.0 300.0 200.0 iso 1.0 1.0 1000.0
```

其中：

- `temp 300.0 300.0 200.0` 指定温度起点、终点和阻尼时间
- `iso 1.0 1.0 1000.0` 指定各向同性目标压强及其阻尼时间

若采用各向异性或三斜盒子，也可以写成：

```lammps
fix 1 all npt temp 300.0 300.0 200.0 aniso 1.0 1.0 1000.0
```

或

```lammps
fix 1 all npt temp 300.0 300.0 200.0 tri 1.0 1.0 1000.0
```

也可以逐个分量指定，例如：

```lammps
fix 1 all npt temp 300.0 300.0 200.0 \
               x 1.0 1.0 1000.0 \
               y 1.0 1.0 1000.0 \
               z 1.0 1.0 1000.0 \
               couple none
```

这些写法的共同特点是：

1. 输入语法完全围绕 `temp + 压力关键字` 组织
2. 热浴类型默认就是 Nose-Hoover chain
3. 用户无法在原版 `fix nvt`、`fix nph`、`fix npt` 中额外指定 `thermostat`、`barostat`、`integrator`、`seed` 或 `zero`

### 0.5 本文实现与原版三类 fix 的直接对比

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
               zero 1
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
               zero 1
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

- `lan_c1_t` 对应 \(c_1^{\mathrm{temp}}\)
- `lan_c2_t` 对应 \(c_2^{\mathrm{temp}}\)
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

- `omega_mass[i]` 承担有效活塞质量 \(W\) 的角色
- `omega_dot[i] = lan_coeff1 * omega_dot[i] + kicks[i]` 即离散更新式

因此，代码中

$$
\frac{lan\_coeff2}{\sqrt{\omega\_mass[i]}}
$$

正对应公式中的

$$
\frac{c_2^{\mathrm{press}}}{\sqrt{W}}
$$

### 改动 4：在各向同性压浴中引入 \(1/d\) 修正

### 新增功能

`fix_nh_middle` 并未直接照搬标准 MTTK 文献中的 barostat 噪声强度，而是针对 LAMMPS 原版 `FixNH` 的盒子动量分布引入了各向同性情况下的 \(1/d\) 修正。  
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

这一实现对应的理论表达式为：

$$
\mathcal{L}_{B}^{\mathrm{Lan}} \Delta t:\ 
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

正是对该 \(1/d\) 因子的实现。

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

其中所有 MPI 进程使用同一个 \(\xi_{\alpha}\)：

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
    nh_omega_dot_middle();
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

  if (pstat_flag) nh_omega_dot_middle();
}
```

### 公式说明

该实现对应将 NPT 离散推进重新组织为 middle 型算子分裂。  
按当前代码的实际顺序，可将其概括为：

$$
\mathcal{L}_{v_r}\frac{\Delta t}{2}
\mathcal{L}_{v_t}\frac{\Delta t}{2}
\mathcal{L}_{v_t}\frac{\Delta t}{2}
\mathcal{L}_{v_r}\frac{\Delta t}{2}
\mathcal{L}_{v_{\varepsilon}}\frac{\Delta t}{2}
\mathcal{L}_{h,x_r}\frac{\Delta t}{2}
\mathcal{L}_{x_t}\frac{\Delta t}{2}
\mathcal{L}_{B}
\mathcal{L}_{T}
\mathcal{L}_{x_t}\frac{\Delta t}{2}
\mathcal{L}_{h,x_r}\frac{\Delta t}{2}
$$

其中：

- \(\mathcal{L}_{v_t}\) 由 `nve_v()` 实现
- \(\mathcal{L}_{v_r}\) 由 `nh_v_press()` 实现
- \(\mathcal{L}_{v_{\varepsilon}}\) 由 `nh_omega_dot_middle()` 实现
- \(\mathcal{L}_{x_t}\) 由 `nve_x_half()` 实现
- \(\mathcal{L}_{h,x_r}\) 由 `remap()` 实现
- \(\mathcal{L}_T\) 为粒子热浴
- \(\mathcal{L}_B\) 为压浴

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

### 改动 8：增加新的 barostat 半步推进 `nh_omega_dot_middle()`

### 新增功能

为适配当前 middle 路径与 LAMMPS 风格 box momentum 方程，`fix_nh_middle` 没有直接复用原版 `FixNH::nh_omega_dot()`，而是增加了新的 `nh_omega_dot_middle()`。

### 关键代码

```cpp
void FixNHMiddle::nh_omega_dot_middle()
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

因此，`nh_omega_dot_middle()` 是将连续的 box momentum 方程离散为 middle 积分半步推进的核心实现。

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

`zero_flag` 不仅控制是否去除质心随机冲量，还同时控制内部温度 compute 的 `extra/dof` 设置。  
其行为可以概括为：

- `zero 1` 时，保留当前温度 compute 的默认 `extra/dof` 设置
- `zero 0` 时，自动将内部温度 compute 的 `extra/dof` 改为 0

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

但在实际模拟中，\(N_f\) 未必只是简单的平动自由度计数，因为体系还可能叠加其他约束或额外自由度修正。  
因此，`fix_nh_middle` 在这里并不直接假定某个固定的 \(N_f\)，而是通过修改内部温度 compute 的 `extra/dof` 参数来控制温度定义所使用的自由度数。

就实现而言，`apply_zero_dof_mode()` 的作用仅需理解为：

- `zero 1`：保留默认的 `extra/dof`
- `zero 0`：自动执行与 `compute_modify ... extra/dof 0` 等价的处理

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

也可以写成逻辑值 `1/0`，因为底层调用的是：

```cpp
zero_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
```

含义：

- `zero yes` 或 `zero 1`：去除粒子随机项产生的质心净冲量
- `zero no` 或 `zero 0`：保留质心热化

同时，`zero 0` 还会自动把内部温度 compute 的 `extra/dof` 改为 0。  
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
- `zero 1`

### 2.5 使用 middle + 双 Langevin 的输入示例

对于 `fix nvt/mid`，若希望使用 middle + Langevin 温度热浴，可写为：

```lammps
fix 1 all nvt/mid temp 300.0 300.0 200.0 \
               integrator middle \
               thermostat langevin 200.0 \
               seed 123456 \
               zero 1
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
               zero 1
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

原因是 `fix_nh_middle` 已经通过 `apply_zero_dof_mode()` 自动处理了这一点：

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

- `zero 1` 时，保留当前温度 compute 的默认 `extra/dof` 设置
- `zero 0` 时，自动执行与 `compute_modify ... extra/dof 0` 等价的处理

### 2.9 推荐的输入模板

若目标是使用本文实现的核心功能，即 middle + Langevin + 去质心随机冲量，推荐模板如下：

```lammps
fix 1 all npt/mid temp Tstart Tstop Tdamp \
               iso Pstart Pstop Pdamp \
               integrator middle \
               thermostat langevin Tdamp_lan \
               barostat langevin Pdamp_lan \
               seed 123456 \
               zero 1
```

若希望保留质心热化，并同时将内部温度 compute 的 `extra/dof` 设为 0，则可写为：

```lammps
fix 1 all npt/mid temp Tstart Tstop Tdamp \
               iso Pstart Pstop Pdamp \
               integrator middle \
               thermostat langevin Tdamp_lan \
               barostat langevin Pdamp_lan \
               seed 123456 \
               zero 0
```

这两种写法的唯一区别是：

- `zero 1`：去除随机项带来的质心净冲量
- `zero 0`：不去除该质心净冲量，并自动执行与 `compute_modify ... extra/dof 0` 等价的处理
