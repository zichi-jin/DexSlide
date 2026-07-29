---
document_type: ai_handoff
topic: DexAlign offline hand calibration under observation-model mismatch
updated_at: 2026-07-22
status: redesign_planned
language: zh-CN
intended_reader: external_ai_research_assistant
standalone: true
---

# DexAlign 问题说明

## 1. 背景

我们有一套手部外骨骼手套。

它的关键特征是：

1. 手套背面有机械结构，能够读取手指关节角，当前一共输出 `20 DoF` 编码器角度。
2. 手套背面还刚性固结了一个 marker body，可以在每一帧稳定求出它在相机坐标系或世界坐标系下的 `6 DoF` 位姿。
3. 同时还有一套 RGB-D 观测，可以通过 `MediaPipe + depth` 得到手部 `21` 个 3D 关键点。

因此，每一帧我们实际上同时拥有三类观测：

1. 手套编码器角度。
2. marker body 位姿。
3. RGB-D 反投影后的手部 3D 关键点。

我们想做的事情不是实时手势识别，而是做一套离线个体化标定，让这副手套对应的手部运动学模型更接近真实佩戴者。

---

## 2. 为什么会有 DexAlign

在 DexAlign 之前，我们已经有两类粗初值：

1. 一个从静态照片估出来的初始 hand skeleton。
2. 一个从 marker 和 palm 几何近似推出来的 `marker-to-hand` 初始外参。

这两类初值都能用，但都不够好。

问题在于：

1. 仅靠照片得到的 skeleton 缺少真实佩戴状态下的 3D 约束。
2. 仅靠 palm 局部几何去算 `marker-to-hand`，只能给根部一个粗对齐，无法同时修正整只手的骨长和掌部布局。
3. 真正最有信息量的数据，其实是“戴着手套运动时”同步采到的编码器、marker pose 和 RGB-D 3D keypoints。

于是自然会想到一个离线联合标定思路：

1. 用手套编码器驱动前向运动学，预测当前这只手在某个 hand frame 下的 `21` 个关键点。
2. 用 marker pose 把这只“预测的手”放到相机系或世界系。
3. 让预测关键点去拟合 RGB-D 观测关键点。
4. 在多帧数据上同时优化：
   - 个体化 skeleton 参数。
   - `marker-to-hand` 的刚体外参。

这套思路就是 DexAlign 的来源。

---

## 3. 问题是怎么被数学化的

### 3.1 坐标系定义

记：

1. `M`：marker body 坐标系。
2. `H`：hand root 坐标系。
3. `C_t`：第 `t` 帧的观测坐标系，可以是相机系，也可以是 world，只要 marker pose 和 3D keypoints 在同一坐标系即可。

### 3.2 每帧已知量

对每一帧 `t`，我们已知：

1. `^C_t T_M`：marker body 在观测系下的 `6 DoF` 位姿。
2. `q_t ∈ R^20`：手套编码器角度。
3. `y_t ∈ R^(21x3)`：RGB-D 得到的 `21` 个 3D 关键点。
4. 每个关键点还带有 confidence 和 depth-valid mask。

### 3.3 要优化的未知量

我们当前把未知量分成两部分：

1. `theta_skeleton`：个体化 skeleton 参数。
   - 主要包括各指骨长。
   - 以及 palm 基点布局。
2. `^M T_H`：从 marker body 到 hand root 的常量刚体变换，`6 DoF`。

### 3.4 前向预测公式

给定某一帧的编码器角度 `q_t`，前向运动学会在 hand frame 下输出 `21` 个预测关键点：

`x_t = f(q_t; theta_skeleton)`

再利用 marker pose 和待优化的外参，把它们变到观测系：

`y_hat_t = ^C_t T_M · ^M T_H · x_t`

于是就得到一个最直接的多帧拟合目标：

`min_{theta_skeleton, ^M T_H} Σ_{t,i} w_{t,i} || y_hat_{t,i} - y_{t,i} ||^2`

其中 `w_{t,i}` 由以下因素组成：

1. 关键点置信度。
2. depth 是否有效。
3. 一个人工设定的类别权重。

当前实现使用的是 robust nonlinear least squares，单位统一为 `mm`。

---

## 4. 这套算法为什么一开始看起来是合理的

这个建模直觉上并不离谱，原因有三个：

1. marker body 是刚性固连在手套上的，因此它给了一个很强的全局位姿锚点。
2. 编码器角度提供了各指姿态，因此 hand pose 不需要完全从视觉反推。
3. 多个姿态序列会共同约束同一套个体参数，理论上足以把粗初值精炼成更准确的 personalized model。

换句话说，DexAlign 原本想做的是：

“利用视觉给的 3D 关键点，去反标定一只佩戴状态下的个体化 hand model，而不是只靠照片或单帧 palm 对齐。”

---

## 5. 这不是纸上方案

这套方法已经被做成可运行的离线流程，所以现在的问题不是“能不能做出来”，而是“做出来以后，真实数据为什么仍然明显不对”。换句话说，当前阻碍已经从工程可用性，转移到了观测模型是否正确这个层面。

---

## 6. 真实数据上的现象

当前最典型的一组真实数据大致有这些特征：

1. 总帧数约 `600`。
2. 每帧有效 3D keypoints 平均约 `20.92 / 21`，说明 hand landmarks 并不是大面积丢失。
3. marker 平均重投影误差约 `0.84 px`，说明 marker 检测本身不算明显崩坏。
4. 但 marker 可见数量波动不小，只有 `1~2` 个 marker 的帧不少。

在这组真实数据上：

1. 初始平均误差约 `36.89 mm`。
2. 联合优化后平均误差约 `19.09 mm`。

这说明两件事：

1. 优化器不是完全没工作，因为误差确实从 `36.9 mm` 降到了 `19.1 mm`。
2. 但 `19 mm` 这个量级肉眼看仍然很离谱，不能算成功标定。

更关键的是，还做过一个诊断：

如果不强迫所有帧共享同一个 `^M T_H`，而是允许“每一帧都单独找一个自由刚体”去把 FK 手模型对齐到该帧的 3D keypoints，则平均误差还可以进一步降到约 `12.37 mm`。

这说明：

1. 当前优化并非完全失败。
2. 但“常量 `marker-to-hand` 外参 + 现有观测模型”没有把真实误差结构解释对。

---

## 7. 为什么现在会卡住

我们目前的判断是：主问题更像建模错误，而不是简单调参问题。

### 7.1 第一层矛盾：FK 点和 RGB-D 点的语义并不一致

当前 hand skeleton / FK 更接近：

1. 关节旋转中心。
2. 骨架节点。
3. 手内部的运动学参考点。

但 RGB-D + MediaPipe 给出的 3D keypoints 更接近：

1. 手表面点。
2. 深度图上的可见表面位置。
3. 带有 surface semantics 的观测点，而不是 internal joint center。

这带来一个第一性问题：

当前目标函数默认假设

`kinematic joint center ≈ observed 3D surface landmark`

但这个假设大概率不成立。

尤其是手指弯曲时，表面点相对骨骼旋转中心会产生系统偏移，看起来就像“骨段长度在变”。因此优化器实际上是在用“静态骨长 skeleton”去拟合“表面运动轨迹”，理论上先天不匹配。

### 7.2 第二层矛盾：marker 在手背，hand root 却不在手背

marker body 固结在外骨骼手套背面，这在工程上很合理，因为容易看见、容易解位姿。

但这也意味着：

1. marker 的刚体坐标系并不天然等同于 wrist 关节中心。
2. 也不天然等同于 palm 内部的 hand root 定义。
3. 从 marker 到 hand root 的刚体外参，本身就跨越了“手背刚体结构”和“手部内部运动学参考系”这两套不同语义。

因此，只要 palm / wrist 的 3D 观测带一点系统偏差，或者 marker 姿态带一点角度误差，优化器就会把这些误差错误地解释成 `^M T_H` 的变化。

物理上，`^M T_H` 应该几乎是常量；但从数据反推出来的 implied `marker-to-hand` 却出现了明显跨帧漂移。这更像是观测偏差被吸收到外参里，而不是真的发生了那么大的相对滑动。

### 7.3 第三层矛盾：palm / wrist / thumb base 这些点最难解释

实际误差模式也支持上面的判断。

在当前结果里，通常最差的一类点不是所有点平均一起坏，而是：

1. `wrist`
2. `thumb_base`
3. 一些 palm-like 点

这些点恰好最容易同时受到以下因素影响：

1. 表面点与内部 hand root 语义错位。
2. 手背结构、手掌表面和 marker 安装位置之间的偏差。
3. depth 视角变化带来的局部系统误差。

### 7.4 第四层矛盾：缺少先验时，优化器会学到“补偿解”

当前优化里，skeleton 参数是自由调的，但并没有足够强的形状先验去约束：

1. 骨长应该落在哪个合理范围内。
2. palm 布局应该满足怎样的解剖结构。
3. `marker-to-hand` 不应为了补偿视觉噪声而乱漂。

于是优化器虽然能继续降误差，但它学到的未必是真实个体手参数，更可能是：

“一组用来补偿观测模型错误的伪参数。”

---

## 8. 当前有哪些证据支持这个判断

### 8.1 权重不是主因

我们已经试过调整关键点权重。结果会有变化，但量级不大：

1. 把权重全部设成 `1`，平均误差仍在大约 `18 mm`。
2. 换别的权重表，可能变成 `19 mm` 或 `20+ mm`。

所以“再调权重”不是主方向。

### 8.2 它不像一个一眼能修掉的低级 bug

目前没有证据表明问题主要来自以下典型错误：

1. `deg/rad` 用错。
2. `marker-to-hand` 变换方向写反。
3. 外参旋转根本没进入优化。

原因很简单：如果这些地方有致命错误，通常不会出现“误差能稳定从 `36.9 mm` 降到 `19.1 mm`”这种现象。

### 8.3 观测骨段长度本身就在抖

直接从 RGB-D 3D keypoints 去计算一些相邻骨段长度，会发现它们跨帧波动通常有 `4~9 mm` 的标准差。

这说明观测本身就不是“稳定的 joint-center measurement”，而更像带明显表面噪声和视角偏差的 surface landmarks。

### 8.4 per-frame 自由刚体还能明显继续降误差

如果允许每一帧都自由找一个刚体去对齐，而不是共享常量 `^M T_H`，误差还能从 `19.09 mm` 继续降到约 `12.37 mm`。

这进一步说明：

1. 当前固定外参假设没有被观测严格支持。
2. 或者说，观测噪声和语义错配大到足以把“本应固定的外参”扭曲成一个时变补偿项。

---

## 9. 当前最核心的判断

我们现在更倾向于下面这个表述：

DexAlign 的主要困难，不是优化器不收敛，也不是权重没调对，而是当前模型少了一层从“真实手部运动学状态”到“RGB-D 表面关键点观测”的 observation layer。

也就是说，问题更像这样：

`latent hand kinematics -> bone / segment pose -> surface landmark location -> RGB-D observation`

而不是现在这种过于直接的：

`latent hand kinematics -> observed 3D keypoint`

如果这层 observation model 不补上，那么优化器只能被迫通过下面几种错误方式去解释数据：

1. 改骨长。
2. 改 palm 几何。
3. 改 `marker-to-hand` 外参。

于是它就会稳定地产生一个“数值上能降误差，但物理上不可信”的补偿解。

---

## 10. 希望外部 AI 重点帮助回答什么

我们希望查的是：过去的 CV、hand model fitting、RGB-D 手部重建、运动学标定领域里，有没有成熟方法专门处理这里这种结构性错配。

重点问题如下。

### 10.1 关于观测模型

1. 当运动学模型的节点代表 joint centers，但观测来自手表面 landmarks 时，通常怎么建模二者之间的系统偏移？
2. 常见做法是不是给每个 landmark 加 bone-local offset、surface anchor、skinning weight，或者直接引入 mesh / MANO 一类统计手模型？
3. 这种问题更接近 MANO fitting、articulated ICP、bundle adjustment with latent offsets，还是 factor graph calibration？

### 10.2 关于 marker-to-hand 外参

1. 当刚性 marker 固定在手套背面，而 hand root 是一个内部运动学参考系时，文献里通常如何标定这类 extrinsic？
2. `marker-to-hand` 应被当作严格常量，还是当作“弱时变 latent state + 强正则”的量来建模？
3. 有没有现成方法专门处理 soft-tissue / glove compliance 导致的 apparent extrinsic drift？

### 10.3 关于 palm / wrist 观测

1. 对 `wrist`、`thumb_base`、palm-like 点这类高偏差点，成熟方法通常是：
   - 直接删掉
   - 降权
   - 单独建模
   - 还是用 visibility-aware uncertainty model？
2. 当 marker 可见数量变化较大时，是否有更标准的 frame selection 或 uncertainty propagation 做法？

### 10.4 关于整体方案升级

1. 如果只允许做最小增量改造，优先应该补 observation layer，还是先加 skeleton prior？
2. 如果允许中等规模重构，是否应该转向 statistical hand model，再把 glove encoder angles 当 pose constraint？
3. 是否存在适合这种“编码器 + marker pose + RGB-D sparse landmarks”三源联合标定的成熟开源工具链？

---

## 11. 希望外部 AI 输出的答案形式

如果外部 AI 要给建议，最好直接回答下面几项：

1. 这个问题在既有文献里最接近哪一类问题。
2. 是否已有成熟算法、模型或开源实现可以迁移。
3. 如果没有完全对口的方法，最值得借鉴的 observation model 是什么。
4. 如果只能小改当前方案，第一优先级该改哪一层。
5. 如果允许中等规模重构，推荐的一条完整技术路线是什么。

---

## 12. 当前准备实施的重构方向

截至 `2026-07-22`，当前更倾向的重构路线不是继续做“一次性全量联合优化”，而是切成 3 步：

1. `S1`：只戴背板，不戴指套，只学习 palm 上 `wrist -> 五指 base` 的 `5` 条方向。
2. `S2`：完整佩戴，先只学习 `20` 个关节角通道到模型自由度的线性映射，目标只看 `15` 根骨段方向。
3. 最后固定前两步结果，再学习全部长度和 `marker2hand` 平移。

这条路线还有两个关键约束：

1. `marker2hand` 的旋转固定为当前 `initial_guess`，不参与优化。
2. 拇指 `CMC` 插补和相关牵连自由度方案已被放弃，不再作为主方向。

也就是说，当前最值得外部 AI 帮助判断的是：

1. 这类“先方向、后长度”的分阶段手部标定路线是否合理。
2. 第二步用骨段方向校 `20` 个关节通道的做法，在既有文献里是否有更稳妥的变体。
3. 第三步在固定方向后做长度和根平移拟合，是否有更标准的线性或半线性求解方式。

---

## 13. 一句话总结

我们现在遇到的阻碍，本质上是：

一套以“内部关节中心 / 骨架节点”为语义的 hand kinematic model，被直接拿去拟合来自 RGB-D 的“手表面 3D landmarks”；同时 marker 安装在手背、hand root 却是内部运动学参考系，于是观测偏差又进一步污染了 `marker-to-hand` 外参估计。结果就是优化器能降误差，但学到的是补偿解，不是可信的个体化手模型。

---

## 14. 机器可解析摘要

```json
{
  "system": {
    "device": "dorsal exoskeleton glove",
    "signals": [
      "20-DoF glove encoders",
      "rigid marker-body 6DoF pose",
      "RGB-D MediaPipe 21-point 3D hand landmarks"
    ],
    "goal": "offline personalization of hand skeleton parameters and marker-to-hand extrinsic"
  },
  "model": {
    "unknowns": [
      "theta_skeleton",
      "constant marker_to_hand transform"
    ],
    "per_frame_known": [
      "marker pose",
      "encoder angles",
      "21 observed 3D keypoints",
      "confidence and valid mask"
    ],
    "equation": "y_hat_t = T_obs_marker(t) * T_marker_hand * FK(q_t; theta_skeleton)",
    "loss": "robust weighted nonlinear least squares in mm"
  },
  "real_data_result": {
    "frames": 600,
    "initial_mean_error_mm": 36.89,
    "optimized_mean_error_mm": 19.09,
    "per_frame_free_rigid_fit_mean_error_mm": 12.37
  },
  "main_blockers": [
    "FK landmarks are internal joint-center-like nodes, but RGB-D landmarks are surface observations",
    "marker is mounted on the dorsal glove, while hand root is an internal kinematic reference frame",
    "palm/wrist/thumb-base observations carry systematic depth and semantic bias",
    "without stronger priors, optimizer absorbs observation bias into skeleton and extrinsic parameters"
  ],
  "current_judgement": [
    "this is more a modeling problem than a weight-tuning problem",
    "the missing observation layer is likely first-order",
    "constant marker-to-hand is physically plausible, but appears frame-varying because errors are absorbed into it"
  ],
  "what_to_search": [
    "surface landmark observation models for articulated hand kinematics",
    "bone-local landmark offsets or skinning for RGB-D hand fitting",
    "marker-to-hand extrinsic calibration under glove compliance or soft-tissue artifact",
    "joint optimization of glove encoders, rigid marker pose, and sparse RGB-D hand landmarks"
  ]
}
```
