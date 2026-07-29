# DexAlign：外骨骼手套的多源个体化手部运动学标定

## 摘要

DexAlign 研究的是一种带有外骨骼机构的手套的离线个体化标定问题。手套机构位于手背侧，并刚性固结一个 marker body，因此可以逐帧获得 marker 在相机坐标系中的位姿；手套同时输出 20 个编码器关节角；RGB-D 相机则通过 MediaPipe 和深度反投影提供 21 个手部 3D 关键点。

目标不是识别手势，而是估计一套能够代表实际佩戴者的手部运动学模型，使编码器驱动的前向运动学结果与视觉观测一致。DexAlign 先后经历了两种主要形式：

1. DexAlign 1.0：把骨架参数和 marker-to-hand 外参放在一个多帧点位目标中联合优化。
2. DexAlign 2.0：把掌部方向、关节角映射、骨段长度和外参平移拆成三个阶段，以降低参数耦合。

实验表明，主要困难不是优化器无法下降，也不是简单的 loss 权重选择问题，而是运动学模型节点与 RGB-D 表面观测之间存在语义错配。优化器可以降低数值误差，但容易通过改变骨长、掌部几何或关节映射生成补偿解，而不一定得到物理上可信的个体化手模型。

---

## 1. 研究背景

### 1.1 硬件与观测

系统由三类同步信息组成：

1. **外骨骼手套编码器**：输出 20 维关节角，记为 `q_t ∈ R^20`。
2. **刚性 marker body**：安装在手套背面，可逐帧求得其 6 DoF 位姿。
3. **RGB-D 手部观测**：MediaPipe 提供 21 个语义关键点，深度图提供每个点的相机系三维坐标、置信度和深度有效性。

marker 并不位于手部内部的 wrist 或关节旋转中心，而是位于手背外骨骼上。它提供的是一个稳定的刚体运动锚点，但 marker 坐标系与 hand root 坐标系之间存在一个需要标定的固定外参。

### 1.2 初始模型

标定开始前已有两类粗初值：

1. 一个 compact hand skeleton，包括 palm 顶点和各手指骨段长度。
2. 一个 marker-to-hand 初始刚体变换。

compact skeleton 的运行时语义为：

1. thumb 由 metacarpal、proximal、distal 三段组成。
2. index、middle、ring、pinky 各由 proximal、middle、distal 三段组成。
3. palm 中包含 wrist、thumb_base、四指 MCP 等基点。

---

## 2. 统一数学建模

记：

- `M`：marker body 坐标系。
- `H`：hand root 坐标系。
- `C_t`：第 `t` 帧相机坐标系。
- `^C_t T_M`：第 `t` 帧 marker 在相机系中的位姿。
- `^M T_H`：marker 到 hand root 的刚体外参。
- `θ`：skeleton 几何参数。
- `FK(q_t; θ)`：在 hand 系中由编码器角驱动得到的 21 个预测关键点。
- `y_{t,i}`：第 `t` 帧第 `i` 个 RGB-D 关键点观测。

最直接的预测公式是：

```text
x̂_{t} = FK(q_t; θ)
p̂_{t,i} = ^C_t T_M · ^M T_H · x̂_{t,i}
```

在 DexAlign 1.0 中，基本目标为：

```text
min_{θ, ^M T_H} Σ_t Σ_i ρ(
    w_{t,i} ||p̂_{t,i} - y_{t,i}||²
)
```

其中 `w_{t,i}` 由 MediaPipe confidence、depth-valid mask 和关键点类别权重构成，内部长度单位统一为 `mm`。无效或非有限点不进入有效残差。

这个模型隐含了一个很强的假设：

```text
运动学模型中的关键点节点 ≈ RGB-D 观测到的 MediaPipe 关键点
```

后续实验表明，这个假设是整个问题的主要风险来源。

---

## 3. DexAlign 1.0：联合点位优化

### 3.1 设计动机

最初的思路是利用同步数据反向个体化 skeleton：

1. 编码器角度提供姿态驱动。
2. marker pose 提供从手套到相机系的刚体定位。
3. RGB-D 关键点提供三维几何目标。
4. 多帧共享同一套 skeleton 和 marker-to-hand 参数。

因此可以在一条目标函数中同时估计：

1. 各手指骨段长度。
2. palm 基点位置。
3. marker-to-hand 的平移和旋转。

### 3.2 联合优化的优点

DexAlign 1.0 的优点是模型简单、实现直接，并且确实能够从较差初值降低点位误差。它把视觉和编码器放进同一个坐标链路，具有明确的几何解释，而不是单独拟合某个经验映射。

### 3.3 联合优化的结构性缺陷

联合优化存在多个不可忽略的耦合：

1. palm 朝向误差可以由 marker-to-hand 旋转补偿。
2. marker-to-hand 平移误差可以由 wrist 或 palm 点偏移补偿。
3. 关节角映射误差可以由骨长改变补偿。
4. 视觉表面点与内部关节中心的偏差可以被骨长、掌点和外参共同吸收。

因此，一个数值上更低的目标值不等价于更真实的骨架。优化器可能学到的是一组补偿参数，而不是佩戴者真实的个体结构。

### 3.4 观测模型错配

当前 compact skeleton 更接近：

1. 关节旋转中心。
2. 骨架节点。
3. 手部内部的运动学参考点。

而 MediaPipe + depth 关键点更接近：

1. 手表面上的可见点。
2. 深度图中某个局部表面位置。
3. 受遮挡、视角和皮肤几何影响的视觉 landmark。

手指弯曲时，表面点相对内部旋转中心的偏移会变化。于是一个静态骨长模型可能观察到类似“动态骨长”的现象。单纯继续优化静态长度，无法从根本上解释这种差异。

---

## 4. DexAlign 2.0：分阶段标定

DexAlign 2.0 的核心思想是把以下问题拆开：

1. palm 基点朝向。
2. 编码器通道到模型自由度的映射。
3. 骨长和 marker-to-hand 平移。

分阶段并不消除观测模型错配，但减少了不同参数族之间互相吞误差的自由度。

### 4.1 固定约定

DexAlign 2.0 固定：

1. 内部使用 21 个关键点。
2. 内部长度单位使用 `mm`。
3. marker-to-hand 的旋转 `R_MH` 固定为初始外参中的旋转。
4. 只有 marker-to-hand 的平移在第三阶段优化。
5. 不再使用 thumb CMC 牵连插补线。
6. S1、S2 的帧先合并为统一 frame pool，再按每帧具备的模态决定其可参与的阶段。

### 4.2 Step 1：palm 五条射线方向

S1 原本设计为只穿戴手套背板、不戴关节指环，因此没有可靠的关节角，但有 marker pose 和 MediaPipe 21 点。

五个 palm base 点为：

1. thumb_base。
2. index_mcp。
3. middle_mcp。
4. ring_mcp。
5. pinky_mcp。

每帧先用固定的 `R_MH` 将观测变换到 hand frame，然后计算：

```text
u_{t,i} = normalize(p_{H,t,i} - p_{H,t,wrist})
```

对每个 base 点做跨帧方向统计，得到 `L_1` 的五条 palm 射线。初始 skeleton 中的五个 wrist-to-base 长度保持不变，因此 Step 1 只改变方向，不改变长度。

这一阶段当前实现主要是加权方向平均和方向约束，而不是完整的普通 nonlinear least-squares。由此产生一个重要性质：对于某个固定关键点类别，若其权重只是在所有帧中乘同一个常数，那么归一化方向平均后这个常数会抵消。因此 Step 1 的类别权重并不能像直觉中那样改变结果。

### 4.3 Step 2：20 个关节通道的 affine 映射

对每个编码器通道使用：

```text
q̂_j = a_j q_j + b_j
```

共 40 个待优化参数。每帧从 MediaPipe 点相减并单位化得到 15 根观测骨段方向，预测侧则用 `L_1` 和 `q̂` 做前向运动学得到 15 根模型骨段方向。

目标形式为：

```text
min Σ_t Σ_k ρ(
    ||d̂_{t,k}(a,b) - d_{t,k}||²
)
  + Σ_j λ^a_j(a_j - 1)²
  + Σ_j λ^b_j b_j²
```

thumb 与后四指允许使用不同正则强度，因为 thumb 的机构和安装关系明显更特殊。

这种设计的优点是把角度映射和骨长分离；但它仍然存在两个问题：

1. 多个 thumb MCP 角度可能通过相互补偿得到相似的方向残差。
2. 方向残差只约束姿态，不直接约束由这些角度产生的可达工作空间。

### 4.4 Step 3：骨长和 marker-to-hand 平移

Step 3 固定 Step 1 的 palm 方向和 Step 2 的关节 affine 映射，只优化：

1. wrist 到五个 base 点的长度。
2. thumb 三段长度。
3. 四指各三段长度。
4. marker-to-hand 平移 `t_MH`。

预测点为：

```text
p̂_{C,t,i} = ^C_t T_M · (R_MH^0, t_MH) · FK_i(q̂_t; L)
```

同时利用 wrist 观测从每帧 marker pose 中计算平移样本：

```text
t_{MH,obs}(t) = R_{CM}(t)^T (p_{C,t,wrist} - t_{CM}(t))
```

Step 3 的目标同时包含关键点点位残差和平移样本残差。

后来加入了一个默认关闭的实验开关：允许 thumb_base 严格绕 hand frame 的 x 轴转动，并让 pp/ip/dp 的局部 thumb frame 同步转动。它不是自由的三维 `thumb_xyz`，也不改变 Step 1、Step 2。

---

## 5. 数据采集与工程有效性

### 5.1 采集质量是算法输入的一部分

DexAlign 的每帧数据至少包含：

1. marker pose。
2. 编码器角度或明确的缺失标记。
3. 21 个 MediaPipe 3D 点。
4. 每点 confidence。
5. depth-valid mask。

采集工具后来增加了：

1. RGB 相机实时画面和 MediaPipe overlay。
2. marker pose、wrist 和 21 点的 3D plot。
3. `SPACE` 全局采集开关。
4. 采集过程中的串口有效性检查。
5. 全局通信配置，避免每个脚本自行猜测 glove port。

这些功能不是附属 UI，而是标定实验的有效性控制。曾经出现过这样的数据污染：采集 S2 时没有显式指定 glove port，脚本自动选择了一个存在但没有有效数据的串口；UI 画面使用了另一条有效通信链路，因此视觉上看似手套正常跟随，但写入数据集的编码器却长期停留在默认 A-pose。这样的数据会让优化过程看起来像算法失败，实际上是输入模态失配。

### 5.2 S1 与 S2 的真实角色

S1 和 S2 是采集语义上的数据集，而不是两条互斥的优化流水线：

- S1：主要提供无关节角的 palm 方向和 wrist 平移观测。
- S2：提供关节角、骨段方向和点位观测。
- 合并后：每帧根据 `q`、wrist、palm、marker 等模态是否存在参与相应目标。

因此，不能简单用“数据集名称”决定一帧是否参与某阶段，必须按帧的实际模态 gating。

---

## 6. 权重与优化行为的实证诊断

### 6.1 权重改变小，不等价于权重没有接入

当前实现中，权重会以不同方式进入三个阶段：

1. Step 1：进入方向加权平均，但固定类别常数在归一化后会部分或完全抵消。
2. Step 2：进入每根 segment 方向残差，同时正则项单独拼接。
3. Step 3：直接乘在毫米级 point residual 上。

因此不能只看最终未加权的 `mean-keypoint-error-mm` 判断权重是否生效。

### 6.2 Step 2 的样本数稀释与 Huber 影响

以约 600 帧 S2 为例：

```text
方向数据残差数量 ≈ 600 × 15 × 3 = 27000
joint scale/bias 正则残差数量 = 40
```

在一次真实结果诊断中：

1. 数据项普通 L2 cost 约为 `404`。
2. 正则项普通 L2 cost 约为 `841`。
3. 使用 `Huber f_scale=0.08` 后，数据项 cost 约为 `111`。
4. 正则项 cost 约为 `11.6`。

这说明高权重正则虽然进入了 residual vector，但由于大量正则残差落入 Huber 线性区，实际影响被显著压缩；同时未经按样本数归一化的数据项会随着帧数增加而增强。

### 6.3 Step 3 的权重结构

Step 3 使用毫米点位残差，且报告中的平均误差是未加权几何距离。当前某次结果中，thumb_tip 权重为 10，它单独贡献了约 `1.87×10^6` 的 Huber cost，约占关键点目标的 72%；thumb_base 权重为 0.05，几乎被关闭。

因此不同权重组合看起来“结果差不多”，可能来自：

1. 目标函数已经被 Huber 转成近似 L1 行为。
2. 某些骨长在几何上由多个关键点共同决定，权重变化只改变冲突点之间的折中。
3. 最终报告使用未加权均值，隐藏了 weighted objective 的变化。
4. 全部权重同时乘同一个常数时，若大部分残差都在 Huber 线性区，目标函数近似整体缩放，最优点几乎不变。

结论是：权重调节不是当前首要的模型修复手段。

---

## 7. 拇指问题的专项证据

### 7.1 当前拇指不是真正的“长度不足”

在一组约 600 帧的 S2 数据中，从 MediaPipe 相邻关键点直接测得的 thumb 段长度中位数约为：

| 段 | MediaPipe 中位数 |
|---|---:|
| wrist -> thumb_base | 31.86 mm |
| thumb_base -> thumb_knuckle | 27.30 mm |
| thumb_knuckle -> thumb_ip | 22.31 mm |
| thumb_ip -> thumb_tip | 18.48 mm |
| 四段折线总长 | 100.83 mm |

当前优化 skeleton 的对应长度约为：

| 段 | 模型长度 |
|---|---:|
| wrist -> thumb_base | 29.38 mm |
| thumb_base -> thumb_knuckle | 29.99 mm |
| thumb_knuckle -> thumb_ip | 21.29 mm |
| thumb_ip -> thumb_tip | 20.15 mm |
| 四段折线总长 | 100.81 mm |

模型总长与 MediaPipe 观测几乎相同，且后三段模型总长约 `71.43 mm`，并没有比观测的后三段总长 `68.10 mm` 更短。因此继续人为增加 thumb length 没有数据依据，真正问题更可能是方向、关节零位和工作空间投影。

### 7.2 当前主要嫌疑是 MCP affine 映射

当前一组结果中的 thumb MCP 映射大致为：

```text
mcp_front: scale ≈ 0.772, bias ≈ -30.2°
mcp_back:  scale ≈ 1.882, bias ≈ -32.6°
```

这两个参数会显著改变后三段 thumb 的空间方向。用同一模型和同一平均 S2 姿态进行诊断：

1. 使用当前映射，thumb tip 的 hand-frame `x` 坐标比 index MCP 少约 `5.1 mm`。
2. 只把 mcp_front bias 置零，thumb tip 的 `x` 坐标相对 index MCP 改善约 `8.7 mm`，并越过 index MCP 的 `x` 位置。

这不是证明 bias 必须手工置零，而是说明“继续加长”不是最有效的自由度；MCP 映射会把一条足够长的拇指投向错误方向。

### 7.3 Step 3 的 thumb_base x 旋转效果有限

新增的严格 x 轴 thumb_base 旋转开关在真实数据上得到约 `-1.63°` 的旋转，mean keypoint error 从约 `9.43 mm` 降到约 `9.13 mm`。改进很小，说明主要误差不在 palm thumb_base 的整体 x 轴朝向，而在 thumb MCP 角度映射和固定 thumb frame 语义。

### 7.4 解析逆解的诊断结果

根据当前 thumb frame：

```text
R_thumb = Rx(75°) · Rz(-90° + mcp_back)
          · Ry(90° + mcp_front) · Rx(-5°)
```

第一段 thumb metacarpal 的方向给出两个独立观测自由度。去除固定的 `Rx(75°)` 后，可写出一组解析关系：

```text
v' = Rx(-75°) v
mcp_front = atan2(-v'_z, sqrt(v'_x² + v'_y²)) - 90°
mcp_back  = atan2(v'_y, v'_x) + 90°
```

对当前 S2 做初步逐帧解析和 affine regression，得到：

```text
mcp_front: scale ≈ 0.336, bias ≈ -84.4°, RMSE ≈ 4.9°
mcp_back:  scale ≈ 1.345, bias ≈   2.8°, RMSE ≈ 13.2°
```

该结果没有直接证明解析逆解可以立即替代 Step 2。尤其是 mcp_back 的误差仍很大，说明观测 frame、手指表面点语义、固定 frame 偏置或角度分支之间还存在不一致。但它证明了一个重要事实：thumb MCP 的两个自由度在几何上并非完全不可观测，当前联合方向优化可能把多个因素混在了一起。

---

## 8. 当前理论判断

### 8.1 问题类型

DexAlign 更接近以下问题的交叉：

1. articulated hand kinematic calibration。
2. multi-sensor extrinsic calibration。
3. nonlinear system identification。
4. articulated model fitting with an observation model。
5. robust bundle adjustment with latent kinematic parameters。

它不是单纯的姿态识别，也不是单纯的 skeleton scaling。

### 8.2 缺失的 observation layer

更合理的生成模型应为：

```text
latent hand kinematics
    -> joint centers / bone frames
    -> surface landmark location
    -> RGB-D depth and MediaPipe observation
```

而不是直接假定：

```text
latent hand kinematics -> observed 3D landmark
```

一个最小的 observation layer 可以写成：

```text
ŷ_{t,i} = T_t · (x_{t,i}(q_t, θ) + B_i(q_t, θ) δ_i)
```

其中：

1. `x_{t,i}` 是运动学节点。
2. `B_i` 是对应 bone 或局部 frame。
3. `δ_i` 是从内部节点到视觉表面点的 bone-local offset。

更完整的版本可以使用 articulated mesh、skinning 或统计手模型，让表面 landmark 随姿态产生合理的动态偏移。

### 8.3 marker-to-hand 的物理含义

marker-to-hand 在物理上应近似为常量，因为 marker 刚性固定在手套背板上。但如果 hand root 是内部运动学参考系，而视觉 wrist 是表面点，那么由每帧视觉数据反推的 apparent marker-to-hand 可能表现出时变漂移。这种漂移不一定表示 marker 真在滑动，更可能是 observation bias 被错误吸收到外参中。

因此不应在没有 observation model 的情况下贸然放开每帧外参；更合理的研究方向是：

1. 保持刚体外参常量。
2. 对观测 offset、软组织/手套柔性或视觉误差建模。
3. 如果确有柔性运动，再用低维、强正则的时变状态表示。

---

## 9. 版本演化总结

| 方面 | DexAlign 1.0 | DexAlign 2.0 |
|---|---|---|
| 核心目标 | skeleton 与 marker-to-hand 联合点位拟合 | 方向、关节映射、长度分阶段标定 |
| marker-to-hand 旋转 | 可作为联合优化变量 | 固定为初始外参旋转 |
| marker-to-hand 平移 | 与其他参数联合 | 仅 Step 3 优化 |
| palm 参数 | 与骨长和外参同时优化 | Step 1 单独估计五条 base 射线方向 |
| 关节角映射 | 隐含在 FK 或未单独建模 | Step 2 显式优化 20 组 scale/bias |
| 长度 | 与其他参数耦合 | Step 3 在方向固定后优化 |
| thumb CMC 插补 | 曾尝试引入 | 已删除，不作为主路线 |
| 主要风险 | 补偿解和参数强耦合 | 阶段错误会被后续阶段继承 |
| 主要缺失 | observation model | 仍然缺少 surface observation model |

DexAlign 2.0 不是对 1.0 的简单调参，而是对参数可辨识性和误差传播路径的结构性重构。它减少了联合优化的自由度，但不能自动解决“内部骨架节点与表面 landmark 不同语义”这一第一性问题。

---

## 10. 当前最值得研究的后续路线

### 10.1 最小改造路线

如果不引入完整 mesh，优先顺序应为：

1. 保留固定 `R_MH` 和分阶段 pipeline。
2. 对每个 MediaPipe landmark 引入少量 bone-local offset。
3. 用静态或低维姿态相关 offset 解释表面点与内部节点的差异。
4. 对 Step 2 的 thumb MCP 使用解析逆解或受约束的逐关节回归，减少 mcp_front/back 的互相补偿。
5. 对不同 residual block 按样本数和单位做显式归一化，再选择 Huber 或 Cauchy 等鲁棒函数。

### 10.2 中等规模重构路线

建立一个带 observation layer 的可微 articulated hand model：

1. 编码器角度作为 pose measurement。
2. marker pose 作为 rigid global pose measurement。
3. 视觉 3D landmarks 作为带不确定度的 surface observation。
4. skeleton shape、joint affine mapping、marker extrinsic 和 surface offsets 作为 latent parameters。
5. 使用 bundle adjustment、factor graph 或 differentiable least-squares 联合估计。

### 10.3 完整模型路线

如果允许更大规模重构，可考虑统计手模型或 articulated mesh：

1. 用手模型的 shape 参数表达个体差异。
2. 用 glove encoder 作为 pose constraint，而不是直接把 encoder angle 当作真实关节角。
3. 用 RGB-D 观测约束表面几何。
4. 用 marker body 约束外骨骼刚体姿态。
5. 通过可见性和深度不确定度处理遮挡、反光和表面深度偏差。

这条路线的优势是显式解决 observation-model mismatch，代价是需要 mesh、surface landmark 定义和更多先验。

---

## 11. 结论

DexAlign 的研究问题可以概括为：

> 在一个 marker 刚性固定于手背外骨骼、编码器提供 20 DoF 运动信息、RGB-D 提供 21 个表面 3D landmark 的系统中，如何标定一套可信的个体化手部运动学模型？

DexAlign 1.0 证明了多源联合拟合能够降低观测误差，但也暴露出严重的参数耦合和补偿解问题。DexAlign 2.0 通过“palm 方向 → 关节映射 → 长度与平移”的阶段拆分提高了可解释性和调试性，但当前拇指实验说明：即使总长度已经与 MediaPipe 观测一致，错误的 MCP 角度映射仍然可以让 thumb 在空间中够不到目标。

因此，当前最重要的结论不是“继续把拇指变长”，而是：

1. 先区分骨长误差与姿态投影误差。
2. 先修正 thumb MCP mapping 的可辨识性和 frame 语义。
3. 最终补充从内部运动学节点到 RGB-D 表面 landmark 的 observation layer。

只有这样，优化出的“拇指更长”或“拇指更容易对指”才有可能是模型自然推导出的结果，而不是通过人为权重或手工修改参数制造出来的现象。

---

## 12. 机器可解析摘要

```json
{
  "system": {
    "device": "dorsal exoskeleton glove",
    "rigid_attachment": "marker body fixed to the dorsal glove structure",
    "signals": [
      "20-DoF encoder angles",
      "per-frame marker-body 6-DoF pose",
      "RGB-D MediaPipe 21-point 3D landmarks",
      "confidence and depth-valid masks"
    ],
    "goal": "offline personalized hand-kinematic calibration"
  },
  "dexalign_1": {
    "unknowns": ["skeleton geometry", "marker_to_hand rigid transform"],
    "prediction": "camera_T_marker(t) * marker_T_hand * FK(q(t), skeleton)",
    "objective": "robust weighted multi-frame 3D point fitting",
    "main_failure": "strong coupling and compensation solutions"
  },
  "dexalign_2": {
    "step1": "estimate five wrist-to-palm-base directions from S1",
    "step2": "estimate 20 affine encoder-to-model mappings from 15 segment directions",
    "step3": "estimate skeleton lengths and marker-to-hand translation from point residuals",
    "fixed": ["marker_to_hand rotation", "20-DoF layout", "compact runtime topology"],
    "optional_thumb_extension": "strict thumb-base rotation around hand-frame x axis"
  },
  "empirical_findings": [
    "loss weights are not equivalent to optimizer influence under sample-count imbalance and Huber saturation",
    "thumb total length can already match MediaPipe while workspace reach remains poor",
    "thumb MCP affine mapping is a stronger suspect than total thumb length",
    "the missing first-order component is a surface-landmark observation model"
  ],
  "recommended_research": [
    "bone-local or pose-dependent surface landmark offsets",
    "analytic or constrained inverse calibration for thumb MCP angles",
    "explicit residual-block normalization",
    "articulated mesh or statistical hand model fitting"
  ]
}
```
