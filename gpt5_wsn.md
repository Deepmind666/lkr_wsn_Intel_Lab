# README — TRIM-RPL（Trust + Residual-energy + Link）在 Contiki-NG/COOJA 上的节能与安全联合优化复现指南（含数学公式）

> 目标：用 **公开数据集（Intel Lab）** 驱动的真实业务流量，在 **Contiki-NG/COOJA** 中复现一篇 **WSN 能耗优化 + 路由安全** 的工程化论文结果。对比 **RPL-MRHOF**、信任型 RPL 与 TARF 思想，做**可复现**的实验与消融，达到 **SCI 三区**粒度。

---

## 0. TL;DR（你先跑起来）

```bash
# 1) 拉代码与工具
git clone https://github.com/contiki-ng/contiki-ng.git
git clone https://github.com/contiki-ng/cooja.git
cd cooja && ./gradlew build   # 生成 build/libs/cooja.jar
```

Cooja 构建与使用（含 MSPSim）参考官方 README。([GitHub][1])

```bash
# 2) 下载 Intel Lab 数据（MIT 原始页 或镜像）
# MIT 官方页 (含数据说明与节点坐标等)
# OpenDataLab 镜像
```

MIT 页面提供约 **2.3M 条读数**、**gz 34MB / 解压 150MB**、**54 节点**、**约每 31 秒采样四类传感值**的说明。([MIT数据库组][2])
OpenDataLab 提供中文镜像与直链，信息与 MIT 一致（54 节点、温湿/温度/光/电压、31s 间隔、2004-02-28～04-05）。([OpenDataLab][3])

```bash
# 3) 在 Contiki-NG 例程上启用能耗统计
# 应用 Makefile 中添加：
MODULES += os/services/simple-energest
# 按官方教程编译/运行，后续解析 Energest 输出估算能耗
```

Contiki-NG 的 **simple-energest** / **Energest** 文档与示例见官方教程与 API。([docs.contiki-ng.org][4])

---

## 1. 目录规划（建议你的开源库结构）

```
wsn-trim-rpl/
├── datasets/
│   ├── intel-lab/                 # 原始CSV（MIT或镜像），及清洗脚本
│   └── README.md                  # 数据说明&来源
├── contiki-ng/                    # 上游子模块（或外部路径）
├── cooja/                         # 上游子模块（或外部路径）
├── apps/
│   ├── rpl-udp-sod/               # 在 rpl-udp 基础上加入 SoD/双预测上报
│   └── trim-of/                   # 新的 RPL 目标函数（联合信任-能量-链路）
├── scripts/
│   ├── gen_cooja_topology.py      # 用Intel坐标/统计驱动自动生成Cooja topo
│   ├── analyze_energest.py        # Energest/Powertrace 日志转 mJ、FND/HND
│   ├── attack_scenarios.sh        # 一键注入 rank/黑洞/选择性转发等
│   └── grid_search.sh             # w1:w2:w3、阈值k等超参扫描
├── cooja-sims/
│   ├── baseline-mrhof.csc         # 基线：MRHOF（ETX）
│   ├── baseline-trust.csc         # 基线：文献式Trust-RPL
│   ├── ours-trim-rpl.csc          # 我们的方法（TRIM-RPL）
│   └── attacks/*.csc              # 各类攻击场景
└── README.md
```

---

## 2. 数据集与获取

* **Intel Lab（MIT）**：公开的 WSN 实验室部署数据，**54 mote**，每 **31s** 采样 **温度/湿度/光/电压**；提供节点**平面坐标**与数据文件，便于映射到 Cooja 拓扑与**现实业务流量**重放。([MIT数据库组][2])
* **镜像/备份**：OpenDataLab（中文站）与 Kaggle 转存（可作备选）。([OpenDataLab][3], [kaggle.com][5])

> 我们用 **白天/夜晚**的统计波动驱动 SoD 阈值（见 §4），并用节点坐标生成 Cooja 地图，使链路与负载“像真的一样”。

---

## 3. 仿真环境与基线代码

### 3.1 Contiki-NG 与 Cooja（RPL 首选）

* **Contiki-NG 主库**与**教程**（RPL/6LoWPAN/TSCH 等协议栈、能耗估计工具链）。([docs.contiki-ng.org][6])
* **Cooja** 仿真器（含 MSPSim），使用 `./gradlew build` 生成 `cooja.jar`。([GitHub][1])
* **能耗估计**：**Energest** 与 **simple-energest** 教程与 API（轻量级、软件统计各硬件状态时长）。([docs.contiki-ng.org][4])
* **Powertrace（Contiki 3.x）** 作为参考（旧版工具，理解能耗统计思路）。([GitHub][7], [eistec.github.io][8])

### 3.2 RPL 目标函数基线：**MRHOF（RFC 6719）**

* MRHOF：最小化度量（常用 ETX），并加入**磁滞**减少抖动，是 Contiki-NG 默认/常见配置之一。([rfc-editor.org][9])
* 中文解读可参考社区博文与 RFC 中文翻译。([rfc2cn.com][10], [CSDN博客][11])

### 3.3 安全攻击复现框架

* **rpl-attacks（Contiki/COOJA）**：可一键注入 **rank**、**blackhole**、**选择转发**等攻击。([GitHub][12])
* Contiki-NG 版实现也有民间仓库可参考。([GitHub][13])

### 3.4 信任/安全类对照组（论文/思想）

* **SecTrust-RPL**：典型**信任感知** RPL 的代表性工作。([科学直通车][14], [CoLab][15])
* **TARF**（Trust-Aware Routing Framework）：面向受限设备、**低改造成本**的信任路由思想，便于与现有协议叠加。([weisongshi.org][16], [SpringerLink][17])

> 若你更偏**聚类+寿命**评估路线，可用 **ns-3 + LR-WPAN（802.15.4）+ Energy Framework**；但本文主线聚焦 RPL。([nsnam.org][18])

---

## 4. 我们的方法：**TRIM-RPL**（论文主线，可直接写进“方法”章节）

### 4.1 数据面：**双预测 / Send-on-Delta（SoD）** 降报文（节能的关键）

* 一阶指数平滑/AR(1) 轻量预测：

  $$
  \hat{x}_t=\alpha x_{t-1}+(1-\alpha)\hat{x}_{t-1}
  $$
* **动态阈值**（Bollinger-band 风格）：

  $$
  \delta_t = k \cdot \sqrt{\tfrac{1}{W}\sum_{i=t-W+1}^{t}(x_i-\bar{x})^2}
  $$
* **触发条件**：若 $|x_t-\hat{x}_t|>\delta_t$ 则**上报**，否则本地缓存（汇聚侧用同构模型重建）。阈值 $k$、窗长 $W$ 按 Intel Lab **昼夜统计**自适应。
  近期 SoD/双预测综述与改进显示，可显著降低上报量与能耗。([MDPI][19], [ietresearch.onlinelibrary.wiley.com][20], [科学直通车][21], [SpringerLink][22])

> 实现位于 `apps/rpl-udp-sod/`：应用层对传感输入做预测与阈值判定，再调用 UDP 发送；**不改 MAC/PHY**，工程代价极低。

### 4.2 路由面：**信任-能量-链路**联合父选（安全+可靠）

* 对邻居 $j$ 维护 **Beta 信任**：$\text{Beta}(\alpha_{ij},\beta_{ij})$。成功转发/按时 ACK：$\alpha\!\leftarrow\!\alpha+1$；失败/掉包：$\beta\!\leftarrow\!\beta+1$。

  $$
  T_{ij}=\frac{\alpha_{ij}+1}{\alpha_{ij}+\beta_{ij}+2}
  $$
* 归一化 **剩余能量** $\tilde{E}_j = E_j/E_{\max}$，归一化**链路质量**（ETX 或 LQI）$\tilde{L}_j$。
* **联合代价**替换 MRHOF：

  $$
  C_j = w_1(1-T_{ij}) + w_2(1-\tilde{E}_j) + w_3(1-\tilde{L}_j),\quad
  j^\*=\arg\min_{j\in \mathcal{P}(i)} C_j
  $$

  其中 $\mathcal{P}(i)$ 为满足 RPL **Rank 单调**的候选父集合（避免环路）。信任/能量/链路的三因子度量来自**SecTrust-RPL**与**TARF**思路的轻量融合。([科学直通车][14], [weisongshi.org][16])
* 在 **rank/黑洞/选择转发** 等攻击下，恶意父会因低 $T$ 被压制。攻击由 rpl-attacks 注入。([GitHub][12])

> 实现位于 `apps/trim-of/`：仿 MRHOF 的 Objective Function 接口封装；权重 $w_1\!:\!w_2\!:\!w_3$ 支持运行时配置与网格搜索。

### 4.3 能耗估计（工程可复现）

* **Energest** 统计 CPU/LPM/RX/TX 等状态时长；结合电流表：

  $$
  E=\sum_{s\in\{\text{CPU,LPM,RX,TX}\}} I_s \cdot V \cdot t_s
  $$

  其中 $I_s$ 取自平台手册或 Contiki-NG 示例配置，$t_s$ 由 Energest 时间累积。教程与示例如文档所示。([docs.contiki-ng.org][4])

---

## 5. 如何跑实验（一步步）

### 5.1 准备 Cooja 项目

* 构建 `cooja.jar`，按 README 启动；或在 Contiki-NG 教程中的“**Running a RPL network in Cooja**”逐步操作。([GitHub][1], [docs.contiki-ng.org][23])
* 用 `scripts/gen_cooja_topology.py` 读取 Intel 节点坐标，生成 `.csc` 拓扑（节点位置与无线范围可配）。

### 5.2 业务流量接入（Intel Lab）

* `apps/rpl-udp-sod/` 读取对应节点的时间序列（温/湿/光/电压），按 §4.1 的 **SoD** 规则**自适应上报**；白天阈值放宽、夜间略收紧（可自动根据滚动标准差调整）。数据来源见 §2。([MIT数据库组][2])

### 5.3 开启能耗记录

* 在应用 `Makefile` 中加：`MODULES += os/services/simple-energest`；运行后获取 Energest 日志，按 §4.3 脚本转 **mJ** / **电量**；官方教程提供启用步骤与示例。([docs.contiki-ng.org][4])

### 5.4 对照/消融场景

* **基线**：`cooja-sims/baseline-mrhof.csc`（原生 MRHOF/ETX）；**信任基线**：`baseline-trust.csc`（复现 SecTrust-RPL 的核心打分逻辑或 TARF 式叠加）。([rfc-editor.org][9], [科学直通车][14], [weisongshi.org][16])
* **攻击**：`cooja-sims/attacks/*.csc` 用 rpl-attacks 注入 **rank/黑洞/选择性转发** 等。([GitHub][12])
* **我们的方案**：`cooja-sims/ours-trim-rpl.csc`（SoD + TRIM-OF）。

### 5.5 指标与统计

* **能耗/寿命**：每节点/全网 mJ、**FND/HND/NDL**（首/半/全网死亡时间步）。
* **可靠性**：PDR、端到端延迟、控制开销。
* **安全鲁棒**：攻击下 PDR、错误父选比例、路径稳定性。
* **消融**：仅 SoD、仅 Trust、SoD+Trust；不同 $k$、窗长 $W$；不同 $w_1\!:\!w_2\!:\!w_3$。

---

## 6. 和 ns-3 路线的对照（可作为备选实验）

若你想做**聚类+寿命**的第二条线，可在 **ns-3** 上复现实验：

* **LR-WPAN（802.15.4）** 官方模型文档；**Energy Framework**（BasicEnergySource + RadioEnergyModel）；适合评估 **LEACH/HEED+SoD** 对 FND/HND 的影响。([nsnam.org][18])

---

## 7. 论文写作要点（可直接复用的小节结构）

1. **数据驱动**：强调 Intel Lab 的真实性与代表性（时间跨度、节点数、采样间隔）；我们按昼夜/日内统计自适应 SoD 阈值。([MIT数据库组][2])
2. **方法亮点**：**数据面**（SoD/双预测降上报）+ **路由面**（信任-能量-链路联合父选），**互补增益**。参考 SoD/DP 文献支撑“节能有效”。([MDPI][19], [ietresearch.onlinelibrary.wiley.com][20])
3. **安全鲁棒**：对抗 rank/黑洞/选择转发，引用 rpl-attacks 复现流程与 SecTrust-RPL/TARF 理念依据。([GitHub][12], [科学直通车][14], [weisongshi.org][16])
4. **工程可复现**：Cooja/Contiki-NG、Energest 估算、脚本一键实验，文中贴出版本/commit。([docs.contiki-ng.org][4])
5. **评测三板斧**：能耗-寿命、可靠性、在攻防场景下的稳定性曲线；全面消融。

---

## 8. 关键参考与开源链接（按用途分组）

* **Intel Lab 数据集（含坐标/说明）**：MIT 官方页；镜像（OpenDataLab）；Kaggle 转存（备选）。([MIT数据库组][2], [OpenDataLab][3], [kaggle.com][5])
* **Contiki-NG 文档/教程（含 Energest & simple-energest）**：开发文档、Energest 教程、示例 README、API。([docs.contiki-ng.org][6], [Contiki-NG][24])
* **Cooja 仓库**（构建 `./gradlew build`）：([GitHub][1])
* **RPL-MRHOF（RFC 6719）**：原 RFC 与中文解读。([rfc-editor.org][9], [rfc2cn.com][10])
* **RPL 攻击框架**（Contiki/COOJA）：([GitHub][12])
* **信任路由文献**：SecTrust-RPL；TARF（PDF/图书章节版）。([科学直通车][14], [CoLab][15], [weisongshi.org][16], [SpringerLink][17])
* **SoD/双预测 数据降采样**：MDPI Sensors、IET、Springer 改进、最新 SoD-Bollinger 自适应。([MDPI][19], [ietresearch.onlinelibrary.wiley.com][20], [SpringerLink][22], [科学直通车][21])
* **（备选）ns-3 LR-WPAN / Energy**：官方模型文档。([nsnam.org][18])

---

## 9. 复现实验的最小操作清单（可直接贴到项目根 README）

1. **准备环境**：JDK 8+/11、Python 3.9+、GNU 工具链。
2. **拉取上游**：`contiki-ng/`、`cooja/`。构建 `cooja.jar`。([GitHub][1])
3. **下载数据**：`datasets/intel-lab/` 放置 MIT 原始 CSV；脚本会自动校验行数/字段。([MIT数据库组][2])
4. **生成拓扑**：`python scripts/gen_cooja_topology.py --intel datasets/intel-lab --out cooja-sims/...csc`（根据坐标和连通性生成）。
5. **选择方案**：

   * `cooja-sims/baseline-mrhof.csc`（MRHOF/ETX）；([rfc-editor.org][9])
   * `cooja-sims/baseline-trust.csc`（信任基线）；([科学直通车][14])
   * `cooja-sims/ours-trim-rpl.csc`（SoD + TRIM-OF）。
6. **开启能耗记录**：在你的应用 `Makefile` 加 `MODULES += os/services/simple-energest`；跑完用 `scripts/analyze_energest.py` 生成 **mJ/寿命(FND/HND/NDL)**。([docs.contiki-ng.org][4])
7. **攻击实验**：`scripts/attack_scenarios.sh --type blackhole --ratio 0.1`（内部调用 rpl-attacks 模板启动）。([GitHub][12])
8. **网格搜索**：`scripts/grid_search.sh --w "0.5,0.3,0.2" --k 2.0 --W 60`，输出 CSV/图表。
9. **论文图表**：按 `notebooks/plot_results.ipynb`（可选）绘制 **PDR-能耗-寿命-攻击鲁棒**四象限图。

---

## 10. 结果应呈现的“论文式”结论结构

* **节能收益**：SoD/双预测在 Intel Lab 驱动下显著减少上报与 TX 占空比，能耗 **mJ** 下降与寿命（FND/HND）延后；趋势与近年 SoD/DP 文献一致。([MDPI][19], [ietresearch.onlinelibrary.wiley.com][20])
* **安全鲁棒**：攻击下，TRIM-RPL 的 **低 $C_j$** 倾向避开低信任父，PDR 保持稳定；与 SecTrust-RPL/TARF 的理论一致。([科学直通车][14], [weisongshi.org][16])
* **总体优势**：在**能耗-安全**双目标的 Pareto 上，**SoD + TRIM-OF** 同时优于仅 MRHOF 或仅 Trust 的单维改造。
* **工程可行**：全部基于 Contiki-NG/COOJA + Energest，轻量、**3060 即可**；如需更快，可借 4090 做多轮网格搜索。

---

### 附：你可能用到的命令/片段

**启用 Energest（Makefile）**

```makefile
MODULES += os/services/simple-energest
# 运行后解析 simple-energest 输出，按 E = Σ I_s * V * t_s 估算能耗
```

官方教程逐步示例与 `simple-energest` 源码。([docs.contiki-ng.org][4])

**Cooja 构建**

```bash
cd cooja && ./gradlew build
java -jar build/libs/cooja.jar
```

([GitHub][1])

**Intel Lab 原始数据与说明**（含数据体量、坐标、采样周期等）
— MIT 官方页；镜像（OpenDataLab）。([MIT数据库组][2], [OpenDataLab][3])

---

## 11. 免责声明与可拓展项

* 本 README 给出**方法与脚手架**；你可选做第二条 **ns-3** 备线（LEACH/HEED + SoD）以强化“能耗/寿命”故事线。([nsnam.org][18])
* 若写“相关工作”，记得引用 **MRHOF RFC**、**SecTrust-RPL**、**TARF** 与最新 **SoD/DP** 论文，呼应你的设计。([rfc-editor.org][9], [科学直通车][14], [weisongshi.org][16])

---

需要我把上述骨架**初始化成一个可运行的最小仓库**（含 `.csc` 模板、示例 Makefile、Python 解析脚本）吗？我可以直接给你打一个 zip / git 包，开箱即跑。

[1]: https://github.com/contiki-ng/cooja?utm_source=chatgpt.com "GitHub - contiki-ng/cooja: This is the main repository for the Cooja ..."
[2]: https://db.csail.mit.edu/labdata/labdata.html?utm_source=chatgpt.com "Intel Lab Data - Massachusetts Institute of Technology"
[3]: https://opendatalab.com/OpenDataLab/Intel_Lab_Data/download?utm_source=chatgpt.com "数据集-OpenDataLab"
[4]: https://docs.contiki-ng.org/en/develop/doc/tutorials/Instrumenting-Contiki-NG-applications-with-energy-usage-estimation.html?utm_source=chatgpt.com "Instrumenting Contiki NG applications with energy usage estimation"
[5]: https://www.kaggle.com/datasets/divyansh22/intel-berkeley-research-lab-sensor-data?utm_source=chatgpt.com "Intel Berkeley Research Lab Sensor Data | Kaggle"
[6]: https://docs.contiki-ng.org/en/develop/?utm_source=chatgpt.com "Contiki-NG Documentation"
[7]: https://github.com/contiki-os/contiki/blob/master/apps/powertrace/powertrace.c?utm_source=chatgpt.com "contiki/apps/powertrace/powertrace.c at master - GitHub"
[8]: https://eistec.github.io/docs/contiki/a00343.html?utm_source=chatgpt.com "Contiki 3.x: apps/powertrace/powertrace.c File Reference"
[9]: https://www.rfc-editor.org/rfc/rfc6719?utm_source=chatgpt.com "RFC 6719: The Minimum Rank with Hysteresis Objective Function"
[10]: https://rfc2cn.com/rfc6719.html?utm_source=chatgpt.com "RFC6719 中文翻译 中文RFC RFC文档 RFC翻译 RFC中文版"
[11]: https://blog.csdn.net/frank_jb/article/details/54023993?utm_source=chatgpt.com "RFC 6719中文版： The Minimum Rank with Hysteresis ..."
[12]: https://github.com/dhondta/rpl-attacks?utm_source=chatgpt.com "GitHub - dhondta/rpl-attacks: RPL attacks framework for simulating WSN ..."
[13]: https://github.com/Adelsamir01/RPL-attacks?utm_source=chatgpt.com "GitHub - Adelsamir01/RPL-attacks"
[14]: https://www.sciencedirect.com/science/article/pii/S0167739X17306581?utm_source=chatgpt.com "SecTrust-RPL: A secure trust-aware RPL routing protocol for Internet of ..."
[15]: https://colab.ws/articles/10.1016%2Fj.future.2018.03.021?utm_source=chatgpt.com "SecTrust-RPL: A secure trust-aware RPL routing protocol for Internet of ..."
[16]: https://weisongshi.org/papers/zhan11-tarf.pdf?utm_source=chatgpt.com "Design and Implementation of TARF: A Trust-Aware Routing Framework for WSNs"
[17]: https://link.springer.com/chapter/10.1007/978-3-642-11917-0_5?utm_source=chatgpt.com "TARF: A Trust-Aware Routing Framework for Wireless Sensor Networks"
[18]: https://www.nsnam.org/docs/models/html/lr-wpan.html?utm_source=chatgpt.com "18. IEEE 802.15.4: Low-Rate Wireless Personal Area Network (LR ... - ns-3"
[19]: https://www.mdpi.com/1424-8220/21/21/7375?utm_source=chatgpt.com "Evaluation of Deep Learning Methods in a Dual Prediction Scheme to ..."
[20]: https://ietresearch.onlinelibrary.wiley.com/doi/am-pdf/10.1049/cmu2.12262?utm_source=chatgpt.com "A reliable and energy efficient dual prediction data reduction approach ..."
[21]: https://www.sciencedirect.com/science/article/pii/S157087052500215X?utm_source=chatgpt.com "Context-aware adaptive Send-on-Delta for traffic saving in sensor ..."
[22]: https://link.springer.com/article/10.1007/s11276-019-01950-7?utm_source=chatgpt.com "An improved adaptive dual prediction scheme for reducing data ..."
[23]: https://docs.contiki-ng.org/en/master/doc/tutorials/index.html?utm_source=chatgpt.com "Tutorials — Contiki-NG documentation"
[24]: https://g-oikonomou-contiki-ng.readthedocs.io/en/develop/?utm_source=chatgpt.com "Contiki-NG API documentation! — Contiki-NG documentation"
