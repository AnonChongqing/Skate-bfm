# Skate-BFM Latent Flow：完整训练、测试与可调 Q 计算实现 Prompt

> 将本文件完整交给 Codex。不要只实现示例代码或伪代码；需要在现有 Skate-bfm 仓库中完成可运行、可测试、可继续调参的工程实现。

---

## 0. 你的角色与总任务

你正在维护现有仓库：

```text
/home/hu_wenhui/workspace/Skate-bfm
```

大文件和训练结果必须放在：

```text
/63data1/hwh_data/Skate-bfm
```

当前仓库已经具备：

1. 冻结的 BFM0 `FBcprAuxModel`；
2. BFM0 观测构造；
3. BFM0 29D action → HUSKY 23DoF action adapter；
4. HUSKY 滑板物理环境、接触状态、command 和 push/steer/transition/recovery 相关 reward；
5. Stage 01 的 fixed/phase/steer 控制、latent search 和 few-shot latent adaptation；
6. Stage 02 的 skateboard-aware heuristic feedback。

现有关键代码至少包括：

```text
01_bfm0_motion_husky/skate_bfm01/policy.py
01_bfm0_motion_husky/skate_bfm01/control.py
01_bfm0_motion_husky/skate_bfm01/adapters/actions.py
01_bfm0_motion_husky/skate_bfm01/constants.py
01_bfm0_motion_husky/scripts/search.py
01_bfm0_motion_husky/scripts/adapt.py
husky_sim/src/mjlab_husky/envs/g1_skate_rl_env.py
```

先检查仓库实际文件与接口，不要仅依赖本 Prompt 中的路径猜测。若文件名略有变化，应基于现有实现调整，但必须保持本文定义的模块边界。

你的总任务是新增一个**自包含 Stage 03**，实现：

```text
滑板/机器人/接触/目标状态
            ↓
Mode-conditioned Latent Flow Policy
            ↓
低维 flow u_t
            ↓
Latent Mapper
            ↓
BFM latent z_t^+
            ↓
冻结 BFM0 Actor
            ↓
29D action → HUSKY 23D adapter
            ↓
HUSKY 物理与接触反馈
            ↓
Twin Skate Q Critics + Semi-MDP SAC
```

最终交付内容必须覆盖：

- 完整代码目录；
- 可配置 Twin Q 网络；
- 可切换的 Q 输入方式、Q target 计算方式、Q 聚合方式和 Q loss；
- Latent Flow Policy；
- Latent Mapper；
- 高层 macro-step 环境；
- Replay Buffer；
- 同状态多候选 branch 数据采集；
- Offline Q 预训练；
- Flow Policy warm start；
- Online semi-MDP SAC；
- Q 排序测试；
- 完整 rollout 测试；
- 单元测试、集成测试、日志与 checkpoint；
- README 和运行命令。

不要修改或破坏 Stage 01、Stage 02 的现有行为。Stage 03 按仓库当前“每个 numbered stage 自包含”的原则实现，不新建顶层 `common/`。

---

# 1. 需要先完成的代码审计

在写代码前先检查并记录：

1. `BfmPolicy` 的加载、`act()`、`project()`、`infer_tracking()` 与 device/batch 处理；
2. BFM observation 的真实字段、shape 和历史堆叠方式；
3. `Bfm0ToHusky23ActionAdapter` 的 shared joint index、action scale、reference mode 和 num_envs 依赖；
4. HUSKY 的 physics dt、step dt、decimation、vector env 数量；
5. robot、skateboard、feet、marker、wheel、truck 的状态读取 API；
6. `_get_feet_contact_b()`、脚地接触、轮地接触、接触力和 contact phase 的真实接口；
7. push、steer、transition、regularization reward 的计算和日志接口；
8. termination 与 truncation 的区分；
9. 环境 reset、随机化和状态写回接口；
10. 当前 `run.sh`、`activate.sh`、数据路径和 CUDA device 传递方式。

先生成：

```text
03_latent_flow/IMPLEMENTATION_NOTES.md
```

记录：

- 实际路径；
- 实际 shape；
- 复用代码；
- 需要复制到 Stage 03 的代码；
- 无法直接复用的接口；
- 与本 Prompt 假设不一致的地方；
- 最终采取的兼容方案。

不要因为局部接口不一致就停止。基于实际代码完成适配，并在该文件中说明。

---

# 2. 不可改变的核心原则

## 2.1 第一版冻结 BFM0

训练过程中：

```python
bfm_model.eval()
bfm_model.requires_grad_(False)
```

冻结：

- BFM Actor；
- Backward Map；
- Forward Map；
- BFM CPR Critic；
- BFM Auxiliary Critic；
- Discriminator；
- BFM observation normalizer 的参数和统计。

第一版只训练：

```text
Latent Flow Policy π_flow
Twin Skate Q: Q1, Q2
Target Q1, Target Q2 仅做 soft update
可选 entropy temperature α
```

BFM0 只作为固定的低层动作解码器：

\[
a_t^{29}=\pi_{\mathrm{BFM}}(o_t^{BFM},z_t)
\]

任何训练步骤都必须验证 BFM 参数没有 gradient，也没有 optimizer 更新。

## 2.2 不使用原 HUSKY policy 控制机器人

允许使用：

- 原 HUSKY 环境；
- 原 HUSKY reward；
- 原 HUSKY 成功轨迹用于数据或初始化；
- 原 HUSKY phase 标签用于第一阶段 mode conditioning。

不允许：

- 在新系统 rollout 中混入原 HUSKY Actor 的关节动作；
- 用 hardcoded joint target 替代 BFM Actor；
- 在结果中把 HUSKY policy 的成功当作 Latent Flow 的成功。

## 2.3 高低层频率

默认：

```text
HUSKY physics: 200 Hz
BFM Actor: 50 Hz
Latent Flow Policy: 10 Hz
macro_steps K = 5 个 HUSKY/BFM policy steps
```

所有频率必须来自 config，不要散落硬编码。

---

# 3. 新增目录结构

新增：

```text
03_latent_flow/
├── README.md
├── IMPLEMENTATION_NOTES.md
├── configs/
│   ├── base.yaml
│   ├── q/
│   │   ├── full_preview.yaml
│   │   ├── no_preview.yaml
│   │   ├── minimal.yaml
│   │   ├── sac_min.yaml
│   │   ├── sac_mean_minus_std.yaml
│   │   ├── td3_min.yaml
│   │   └── finite_horizon.yaml
│   ├── train/
│   │   ├── collect_branches.yaml
│   │   ├── offline_q.yaml
│   │   ├── bc_flow.yaml
│   │   └── online_sac.yaml
│   └── eval/
│       ├── q_ranking.yaml
│       ├── rollout.yaml
│       └── ablation.yaml
├── skate_bfm_flow/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── enums.py
│   ├── paths.py
│   ├── env/
│   │   ├── macro_env.py
│   │   ├── state_features.py
│   │   ├── reward_adapter.py
│   │   ├── mode_scheduler.py
│   │   ├── snapshot.py
│   │   └── termination.py
│   ├── bfm/
│   │   ├── frozen_policy.py
│   │   ├── batch_action_adapter.py
│   │   ├── latent_mapper.py
│   │   ├── latent_basis.py
│   │   └── action_preview.py
│   ├── models/
│   │   ├── blocks.py
│   │   ├── skate_q.py
│   │   ├── flow_policy.py
│   │   └── normalizers.py
│   ├── q/
│   │   ├── input_builder.py
│   │   ├── aggregators.py
│   │   ├── targets.py
│   │   ├── losses.py
│   │   └── registry.py
│   ├── data/
│   │   ├── replay_buffer.py
│   │   ├── branch_dataset.py
│   │   ├── batch.py
│   │   └── storage.py
│   ├── algorithms/
│   │   ├── collector.py
│   │   ├── offline_q_trainer.py
│   │   ├── behavior_clone.py
│   │   ├── sac_trainer.py
│   │   └── updates.py
│   ├── evaluation/
│   │   ├── q_ranking.py
│   │   ├── rollout.py
│   │   ├── ablation.py
│   │   └── metrics.py
│   └── utils/
│       ├── checkpoint.py
│       ├── logging.py
│       ├── seed.py
│       ├── torch_utils.py
│       └── git_info.py
├── scripts/
│   ├── inspect_stage03.py
│   ├── build_latent_basis.py
│   ├── collect_branches.py
│   ├── train_offline_q.py
│   ├── train_flow_bc.py
│   ├── train_online_sac.py
│   ├── evaluate_q.py
│   ├── evaluate_flow.py
│   └── run_ablation.py
└── tests/
    ├── test_config.py
    ├── test_feature_shapes.py
    ├── test_latent_mapper.py
    ├── test_action_preview.py
    ├── test_q_network.py
    ├── test_q_targets.py
    ├── test_replay_buffer.py
    ├── test_snapshot_roundtrip.py
    ├── test_frozen_bfm.py
    └── test_macro_env.py
```

实际文件可以根据仓库风格小幅调整，但必须保持：

- 模型、Q 计算策略、环境封装、数据、算法和评测分离；
- Q target 计算不能写死在网络类里；
- Q 输入拼接不能散落在 trainer 中；
- 所有关键超参数都可从配置文件修改。

---

# 4. 配置系统要求

使用：

- Pydantic 配置模型进行类型检查和范围检查；
- YAML 配置文件；
- CLI 支持 `--config path.yaml`；
- 支持重复的 `--set key=value` 点路径覆盖，例如：

```bash
--set q.target.aggregation=mean_minus_std \
--set q.preview.enabled=false \
--set latent.flow_dim=8
```

不要引入 Hydra 这类重型框架。若仓库已有统一配置工具则优先复用。

每次运行必须把**最终解析后的完整配置**保存到 run 目录：

```text
resolved_config.yaml
```

基础配置至少包括：

```yaml
experiment:
  name: latent_flow_v0
  seed: 42
  device: cuda:0
  deterministic: false

paths:
  project_root: /home/hu_wenhui/workspace/Skate-bfm
  data_root: /63data1/hwh_data/Skate-bfm
  bfm_model_dir: /63data1/hwh_data/Skate-bfm/models/bfm0
  run_dir: /63data1/hwh_data/Skate-bfm/runs/latent_flow
  dataset_dir: /63data1/hwh_data/Skate-bfm/datasets/latent_flow
  checkpoint_dir: /63data1/hwh_data/Skate-bfm/checkpoints/latent_flow

control:
  bfm_hz: 50
  flow_hz: 10
  macro_steps: 5
  mean_bfm_action: true

mode:
  names: [push, mount, steer, dismount, recover]
  source: fixed_fsm
  use_one_hot: true

latent:
  z_dim: 256
  flow_dim: 16
  step_size: 0.25
  update_type: tangent_residual
  basis_type: fixed_mode_basis
  basis_trainable: false
  project_radius: sqrt_dim

q:
  input_profile: full_preview
  state_profile: privileged
  twin_independent: true
  architecture: multi_branch
  activation: elu
  final_hidden_dims: [512, 256, 128]
  preview:
    enabled: true
    type: first_action_23d
    detach_bfm: true
  target:
    type: sac_td
    aggregation: min
    gamma_macro: 0.99
    bootstrap_on_timeout: true
    entropy_term: true
    uncertainty_beta: 0.5
  loss:
    type: huber
    huber_delta: 1.0
  optimizer:
    type: adamw
    lr: 0.0003
    weight_decay: 0.00001
    grad_clip: 10.0
  target_tau: 0.005

policy:
  architecture: frame_stack_mlp
  frame_stack: 5
  hidden_dims: [512, 256, 128]
  activation: elu
  log_std_min: -5.0
  log_std_max: 1.0
  optimizer_lr: 0.0003

sac:
  batch_size: 512
  replay_capacity: 2000000
  random_steps: 20000
  update_after: 5000
  updates_per_macro_step: 1.0
  learn_alpha: true
  initial_alpha: 0.1
  target_entropy: auto

reward:
  source: husky_existing
  macro_aggregation: discounted_sum
  normalize: false
  clip: null
  gate_progress_by_retention: true

logging:
  tensorboard: true
  jsonl: true
  csv: true
  checkpoint_interval: 10000
  eval_interval: 10000
```

Config schema必须拒绝：

- `flow_hz > bfm_hz`；
- `macro_steps != bfm_hz / flow_hz`，除非明确允许非整除；
- `flow_dim <= 0`；
- `z_dim` 与真实 BFM checkpoint 不一致；
- `twin_independent=false` 用于正式训练；
- 不支持的 Q target、aggregation、loss 或 input profile。

---

# 5. Mode 定义与第一版训练范围

定义：

```python
class SkateMode(IntEnum):
    PUSH = 0
    MOUNT = 1
    STEER = 2
    DISMOUNT = 3
    RECOVER = 4
```

映射当前 HUSKY/Stage 01 状态：

```text
push          → PUSH
push2steer    → MOUNT
steer         → STEER
steer2push    → DISMOUNT
失稳/脱板事件 → RECOVER
```

第一版：

```text
mode.source = fixed_fsm
```

即 mode 由外部 phase/FSM 提供，Flow Policy只学习模式内部的连续 latent flow。

代码接口必须为以后预留：

```text
mode.source = policy_head
```

但本次不要求完成稳定的自主 mode RL。可以实现网络 head 和 supervised mode loss 的基础接口，但默认关闭。

---

# 6. Latent Flow Policy 的输入与输出

## 6.1 Policy 只能看部署可获得信息

Flow Policy observation：

\[
o_t^{flow}=[s_t^{robot,deploy},s_t^{board,estimated},s_t^{contact,deploy},g_t,z_t,m_t,u_{t-1}]
\]

默认包括：

### Robot deployable features

- joint position relative to default：23；
- joint velocity：23；
- base angular velocity：3；
- projected gravity：3；
- root height：1；
- previous HUSKY action：23；
- 可选估计 base velocity：3。

### Board estimated features

- board relative position in robot local frame：3；
- board orientation 6D：6；
- board local linear velocity：3；
- board local angular velocity：3；
- board heading error sin/cos：2；
- robot-board distance：1。

### Contact deployable features

默认训练阶段可使用仿真接触，但字段要对应未来可由足底/视觉/滑板 IMU估计的量：

- left/right foot-board binary：2；
- left/right foot-ground binary：2；
- contact duration：4；
- foot-marker relative error：6。

### Goal

- local goal xy：2；
- distance：1；
- heading error sin/cos：2；
- target speed：1；
- target contact mask：4。

### Latent control

- current z：256，进入网络前除以 `sqrt(z_dim)`；
- current mode one-hot/embedding；
- previous flow：`flow_dim`。

默认不实现 recurrent replay。使用最近5个高层 observation 的 frame stack。必须保留 `temporal_type` 接口，未来可扩展GRU，但本次默认实现 `frame_stack_mlp`。

## 6.2 Policy 输出

Flow Policy输出 tanh-squashed Gaussian：

\[
\mu_t,\log\sigma_t=f_\phi(o_t^{flow})
\]

\[
\epsilon\sim\mathcal N(0,I)
\]

\[
u_t=\tanh(\mu_t+\sigma_t\epsilon)\in[-1,1]^{d_u}
\]

必须正确计算 tanh correction 后的 log probability，用于SAC。

提供：

```python
policy.sample(obs) -> flow, log_prob, mean_flow
policy.act(obs, deterministic: bool) -> flow
```

---

# 7. Latent Mapper

默认：mode-conditioned fixed low-rank basis。

\[
d_t=U_{m_t}u_t
\]

\[
d_t^\perp=d_t-z_t\frac{z_t^\top d_t}{\|z_t\|^2+\epsilon}
\]

\[
z_t^+=\operatorname{Proj}(z_t+\eta d_t^\perp)
\]

\[
\operatorname{Proj}(x)=\sqrt{d_z}\frac{x}{\|x\|+\epsilon}
\]

接口：

```python
class LatentMapper(nn.Module):
    def forward(
        self,
        z_current: Tensor,      # [B, 256]
        mode_id: Tensor,        # [B]
        flow: Tensor,           # [B, flow_dim]
    ) -> LatentMapOutput:
        ...
```

输出结构：

```python
@dataclass
class LatentMapOutput:
    z_candidate: Tensor
    raw_direction: Tensor
    tangent_direction: Tensor
    delta_norm: Tensor
    cosine: Tensor
```

支持配置：

```text
update_type:
  tangent_residual       # 默认
  euclidean_residual     # 消融
  prototype_residual     # 后续/消融
  direct_z               # 仅测试，不作为默认训练
```

## 7.1 Basis 建立

实现脚本：

```text
scripts/build_latent_basis.py
```

支持：

1. 从已有 reward latent/prototype 文件选择种子；
2. 从同一 mode 的 latent 集合做中心化 PCA/SVD；
3. 构造 `[num_modes, 256, flow_dim]`；
4. 对每个 basis 做正交化；
5. 保存：

```text
/63data1/hwh_data/Skate-bfm/latent_basis/<name>.pt
```

同时保存 JSON metadata：

- source files；
- source key/index；
- mode；
- explained variance；
- normalization；
- git commit；
- SHA256。

第一版 basis 不训练。禁止 Flow Policy、basis 和 BFM Actor 三者同时更新。

---

# 8. Macro Environment

实现：

```python
class LatentFlowMacroEnv:
```

接口：

```python
reset() -> FlowObservation
step(flow_action) -> MacroStepResult
```

一次 `step(flow_action)`：

1. 构造当前 `z_candidate`；
2. 在后续 `K=5` 个低层step中保持该latent不变；
3. 每个低层step重新读取BFM observation；
4. 冻结 BFM Actor 输出29D action；
5. Adapter转换23D action；
6. HUSKY执行物理；
7. 累积reward component、termination和诊断量；
8. 生成下一高层状态。

高层reward：

\[
R_t^{(K)}=\sum_{j=0}^{K-1}\gamma_l^j r_{t,j}
\]

其中：

\[
\gamma_l=\Gamma^{1/K}
\]

默认：

```text
Gamma = 0.99
K = 5
gamma_low = 0.99 ** (1 / 5)
```

必须区分：

```text
terminated: 跌倒、严重脱板、任务成功、环境定义的真实终止
truncated: 时间上限或人为评测长度结束
```

默认对 timeout 继续 bootstrap。

---

# 9. State Feature Builder

实现统一的：

```python
class FlowActorFeatureBuilder
class SkateCriticFeatureBuilder
```

禁止在 collector、trainer、evaluator 中重复手写 feature 拼接。

所有位置、速度和朝向尽量转到机器人局部坐标系或滑板局部坐标系。

## 9.1 Critic默认 privileged feature profile

默认维度目标如下，必须从实际环境验证，不能静默错位：

```text
robot:       79
board:       19
contact:     26
goal_mode:   28
z_current:  256
z_candidate:256
flow:        16
prev_flow:   16
latent stats: 2
action preview: 23
```

真实字段不足时不要用假数据伪装。应：

1. 在 `IMPLEMENTATION_NOTES.md` 说明；
2. 调整 config 中实际维度；
3. 保持 feature name 和 mask；
4. 测试 shape。

Contact force使用稳定变换：

\[
\hat F=\frac{\log(1+\operatorname{clip}(F,0,F_{max}))}{\log(1+F_{max})}
\]

旋转使用6D representation，不直接将四元数作为默认神经网络输入。

---

# 10. Action Preview

默认Q输入包含当前候选latent对应的第一步HUSKY 23D动作：

\[
a_t^{preview}=A_{23}(\pi_{BFM}(o_t^{BFM},z_t^+))
\]

实现：

```python
class FrozenBfmActionPreview(nn.Module):
    @torch.no_grad()
    def forward(bfm_obs, z_candidate) -> Tensor:  # [B, 23]
        ...
```

要求：

- 支持 batch replay 数据；
- 不依赖 live env 的 `num_envs`；
- 将现有 adapter 的 joint index、scale、default pose 转成纯 tensor batch module；
- 与现有 live adapter 在同一输入上的结果数值一致；
- BFM 参数和 adapter 参数不接收梯度；
- 通过 config 可关闭preview。

`q.preview.type` 支持：

```text
none
action_29d
action_23d            # 默认
lower_body_12d        # 消融
```

---

# 11. Twin Skate Q 网络

## 11.1 数学定义

两个网络评价同一个任务：

\[
Q_{\psi_1}(c_t,u_t),\quad Q_{\psi_2}(c_t,u_t)
\]

不是一个管push、一个管steer。

默认完整输入：

\[
Q_i(s_t^{robot},s_t^{board},s_t^{contact},g_t,m_t,z_t,u_t,u_{t-1},z_t^+,a_t^{preview})
\]

两个在线Q结构相同、参数完全独立：

```python
q1 = SkateQNetwork(cfg)
q2 = SkateQNetwork(cfg)
```

禁止共享可训练encoder。可共享：

- feature builder；
- frozen BFM；
- frozen adapter；
- non-trainable normalizer统计。

还要有：

```text
target_q1
target_q2
```

target只soft update。

## 11.2 默认多分支结构

```text
Robot Encoder:        robot_dim       → 256 → 128
Board Encoder:        board_dim       → 128 → 64
Contact Encoder:      contact_dim     → 128 → 64
Goal/Mode Encoder:    goal_mode_dim   → 128 → 64
Current-z Encoder:    256             → 256 → 128
Candidate-z Encoder:  256             → 256 → 128
Flow Encoder:         2*flow_dim + 2  → 128 → 64
Preview Encoder:      preview_dim     → 128 → 64
```

融合：

```text
concat
→ 512
→ 256
→ 128
→ scalar Q
```

默认：LayerNorm + ELU。最后一层无激活。

## 11.3 Q Input Profile 必须可切换

实现 `QInputBuilder` 和 registry，不允许在网络中堆叠大量 `if config.xxx`。

支持：

### `minimal`

```text
robot + board + contact + goal_mode + z_current + flow
```

### `candidate`

```text
minimal + z_candidate + latent_stats
```

### `preview`

```text
minimal + action_preview
```

### `full_preview`（默认）

```text
minimal + z_candidate + previous_flow + latent_stats + action_preview
```

统一输出：

```python
@dataclass
class QInputBatch:
    branch_tensors: dict[str, Tensor]
    batch_size: int
```

网络根据 config 中启用的branches构建，不要给禁用分支传全零后仍保留同样参数量。

---

# 12. Q 计算方式必须模块化

这是本任务的重点。将以下内容彻底分开：

1. Q网络结构；
2. Q输入构造；
3. 双Q聚合；
4. TD/监督target计算；
5. loss；
6. reward macro aggregation。

实现抽象接口：

```python
class QValueAggregator(Protocol):
    def __call__(self, q1: Tensor, q2: Tensor) -> Tensor: ...

class QTargetStrategy(Protocol):
    def compute(self, batch, target_q, policy, context) -> TargetOutput: ...

class CriticLoss(Protocol):
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...
```

## 12.1 Q聚合方式

支持：

### `min`（默认）

\[
Q_{agg}=\min(Q_1,Q_2)
\]

### `mean`

\[
Q_{agg}=\frac{Q_1+Q_2}{2}
\]

### `mean_minus_std`

对两个Q：

\[
\mu_Q=\frac{Q_1+Q_2}{2}
\]

\[
\sigma_Q=\sqrt{\frac{(Q_1-\mu_Q)^2+(Q_2-\mu_Q)^2}{2}+\epsilon}
\]

\[
Q_{agg}=\mu_Q-\beta\sigma_Q
\]

`beta`来自config。

### `min_minus_disagreement`

\[
Q_{agg}=\min(Q_1,Q_2)-\beta|Q_1-Q_2|
\]

用于评测或风险敏感消融，默认不用于第一版主训练。

## 12.2 Q target方式

支持以下 strategy，以 registry 字符串选择：

### A. `finite_horizon_return`

用于同状态branch数据的offline Q预训练：

\[
y_t=G_t^{(H)}=\sum_{j=0}^{H-1}\gamma_l^j r_{t+j}
\]

无bootstrap，训练“候选flow短期结果预测器”。

支持 `H` 为：

```text
10 / 25 / 50 个低层step
```

可通过不同dataset或不同head/独立run训练。本次默认先用单scalar Q，每次run选择一个 horizon。

### B. `semi_mdp_td`

不含entropy：

\[
y_t=R_t^{(K)}+(1-d_t)\Gamma Q_{agg}^{target}(c_{t+K},u')
\]

其中 `u'` 可来自 deterministic policy 或 target policy。

### C. `sac_td`（online默认）

\[
u'\sim\pi_\phi(\cdot|o_{t+K}^{flow})
\]

\[
y_t=R_t^{(K)}+(1-d_t)\Gamma\left[Q_{agg}^{target}(c_{t+K},u')-\alpha\log\pi_\phi(u'|o_{t+K}^{flow})\right]
\]

### D. `td3_td`

为后续消融保留：

- deterministic actor；
- target action smoothing；
- no entropy；
- delayed policy update。

可以完成target strategy和配置接口；本次主训练仍使用SAC。

## 12.3 Q loss

支持：

```text
huber      # 默认
mse
mae        # 只用于诊断，不建议正式训练
```

默认：

\[
L_Q=L_{Huber}(Q_1,y)+L_{Huber}(Q_2,y)
\]

必须记录：

```text
q1_loss
q2_loss
q1_mean
q2_mean
target_mean
td_error_abs_mean
q_disagreement_mean
q_disagreement_p95
```

---

# 13. Replay Buffer

实现预分配Tensor replay，不要用百万条 Python dict。

每条macro transition至少保存：

```python
@dataclass
class MacroTransition:
    flow_actor_obs: Tensor
    critic_robot: Tensor
    critic_board: Tensor
    critic_contact: Tensor
    critic_goal_mode: Tensor

    bfm_obs: dict[str, Tensor] | flattened validated structure
    z_current: Tensor
    mode_id: Tensor
    flow: Tensor
    previous_flow: Tensor

    reward_macro: Tensor
    reward_components: Tensor

    next_flow_actor_obs: Tensor
    next_critic_robot: Tensor
    next_critic_board: Tensor
    next_critic_contact: Tensor
    next_critic_goal_mode: Tensor
    next_bfm_obs: ...
    next_z_current: Tensor
    next_mode_id: Tensor
    next_previous_flow: Tensor

    terminated: Tensor
    truncated: Tensor
```

`z_candidate` 和 `action_preview` 默认可在训练时重算，以确保config切换，但提供可选缓存：

```yaml
replay:
  cache_z_candidate: false
  cache_action_preview: false
```

若缓存，必须保存 mapper/basis hash，并拒绝加载不匹配的dataset。

支持：

- uniform sampling；
- mode-balanced sampling；
- success/failure-balanced sampling；
- dataset保存和恢复；
- replay metadata；
- 大数据写入data disk。

默认训练采样：

```text
push       25%
mount      20%
steer      25%
dismount   15%
recover    15%
```

当某模式数据不足时，记录warning并采用可用分布，不要无限循环。

---

# 14. 环境快照与同状态多候选 branch 数据

实现：

```python
class HuskyEnvSnapshot
```

需要保存恢复足以重现下一步动力学的完整状态，包括：

- robot qpos/qvel；
- skateboard qpos/qvel；
- actuator/internal state（若存在）；
- command state；
- phase/mode state；
- episode counters；
- last action；
- contact history buffers；
- transition reference buffers；
- randomization参数；
- 其他会影响下一步的manager buffer。

接触本身一般由 forward 重新计算，不直接写入，但恢复后必须调用正确的写回和forward流程。

实现 roundtrip test：

1. 保存 snapshot；
2. 执行若干step；
3. 恢复；
4. 使用同一action重新执行；
5. 比较robot/board状态与reward。

目标误差：float32合理范围内接近，默认 `atol <= 1e-5`。若Warp/vector env导致无法完全bitwise一致，需要在说明中记录，并给出实际误差。

若现有MJLab接口无法可靠单环境clone：

- 使用并行环境复制相同完整state；
- 或新建isolated evaluator process；
- 不允许用“只重置到大致相同初始姿态”冒充同状态branch。

## 14.1 Branch collector

脚本：

```text
scripts/collect_branches.py
```

流程：

```text
采样一个有效anchor state
→ 复制到N个branch
→ 采样N个flow候选
→ 每个候选执行H低层step
→ 保存return、接触结果、失败标签和final state
```

候选来源可配置：

```text
flow_policy
zero_flow
local_gaussian
uniform_flow
cem_elite
prototype_direction
current_stage01_heuristic
```

默认混合比例：

```yaml
branch_sampling:
  zero_flow: 0.10
  local_gaussian: 0.40
  uniform_flow: 0.10
  prototype_direction: 0.20
  current_stage01_heuristic: 0.20
```

保存：

```text
anchor_id
candidate_id
mode
flow
z_current
z_candidate
finite_horizon_return
reward_components
fall
contact_loss
illegal_contact
board_progress
heading_progress
retention
```

---

# 15. 训练阶段

## Stage A：工程与数据接口验证

运行：

```bash
source activate.sh
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/inspect_stage03.py \
  --config 03_latent_flow/configs/base.yaml
```

输出：

- feature name、shape、min/max/mean/std；
- BFM z维度；
- BFM action shape；
- adapter action shape；
- reward components；
- mode mapping；
- contact fields；
- termination/truncation；
- data paths；
- no-grad检查。

## Stage B：构造 latent basis

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/build_latent_basis.py \
  --config 03_latent_flow/configs/base.yaml \
  --output /63data1/hwh_data/Skate-bfm/latent_basis/skate_mode_basis_v0.pt
```

输出basis和metadata，并生成每个mode的数值检查报告。

## Stage C：采集同状态branch数据

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/collect_branches.yaml \
  --set branch.num_anchors=5000 \
  --set branch.candidates_per_anchor=16
```

先支持小规模smoke：

```text
10 anchors × 4 candidates
```

再运行正式采集。

## Stage D：Offline Q预训练

先使用：

```text
q.target.type = finite_horizon_return
```

训练两个Q预测同一有限时域return。

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/train_offline_q.py \
  --config 03_latent_flow/configs/train/offline_q.yaml
```

必须：

- train/validation按anchor_id划分，不能让同一anchor的不同candidate跨train/val；
- 记录MSE、Huber、Spearman、Kendall、Top-1 regret、Top-k hit rate；
- 分mode报告；
- 保存best-by-ranking checkpoint和best-by-loss checkpoint；
- 提供小数据过拟合测试。

## Stage E：Flow Policy warm start

对每个anchor选择：

\[
u^*=\arg\max_j G_j
\]

行为克隆：

\[
L_{BC}=\|\mu_\phi(o)-u^*\|^2
\]

可按return softmax做加权回归：

\[
w_j=\operatorname{softmax}(G_j/T)
\]

\[
L_{BC}=\sum_j w_j\|\mu_\phi(o)-u_j\|^2
\]

配置：

```text
bc.target_type = hard_best | soft_weighted
```

脚本：

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/train_flow_bc.py \
  --config 03_latent_flow/configs/train/bc_flow.yaml
```

## Stage F：Online Semi-MDP SAC

使用：

```text
q.target.type = sac_td
q.target.aggregation = min
```

更新顺序：

1. collector执行macro action；
2. transition写入replay；
3. sample batch；
4. 更新Q1、Q2；
5. 更新Flow Policy；
6. 更新alpha；
7. soft update target Q；
8. 定期eval/checkpoint。

Q target：

\[
u'\sim\pi_\phi(o_{t+K})
\]

\[
y=R_t^{(K)}+(1-d)\Gamma\left[Q_{agg}^{target}(c_{t+K},u')-\alpha\log\pi_\phi(u'|o_{t+K})\right]
\]

Critic：

\[
L_Q=L(Q_1,y)+L(Q_2,y)
\]

Actor：

\[
L_\pi=\mathbb E[\alpha\log\pi_\phi(u|o)-Q_{agg}(c,u)]
\]

加入可配置正则：

\[
L_{mag}=\lambda_{mag}\|u\|^2
\]

\[
L_{smooth}=\lambda_{smooth}\|u-u_{prev}\|^2
\]

最终：

\[
L_\pi=\alpha\log\pi-Q_{agg}+L_{mag}+L_{smooth}
\]

第一版默认：

```yaml
policy_regularization:
  flow_magnitude: 0.001
  flow_smoothness: 0.01
```

不要把原 BFM Q、Aux Q、F·z 默认加入Actor loss。为以后消融预留接口，但默认关闭。

---

# 16. Reward 接口

默认复用 HUSKY 现有 reward，但必须通过 `RewardAdapter` 统一输出：

```python
@dataclass
class SkateRewardOutput:
    total_low_level: Tensor
    components: dict[str, Tensor]
    gates: dict[str, Tensor]
```

至少记录：

```text
push_total
steer_total
transition_total
regularization_total
board_progress
heading_progress
foot_board_contact
foot_ground_contact
retention
upright
fall_penalty
illegal_contact
latent_magnitude_penalty
latent_smoothness_penalty
```

若原环境没有某个component，不得伪造。可以由Stage 03 wrapper额外计算，但必须明确命名。

防止reward hacking：

- board velocity/progress奖励必须可配置地受robot-board retention或合法接触gate约束；
- 滑板被踢飞但人板分离不能得到高任务回报；
- transition成功必须同时满足目标脚接触和稳定保持若干step；
- success、fall、contact loss必须记录独立指标，不能只看total reward。

---

# 17. Checkpoint

每个checkpoint保存：

```text
flow_policy
online_q1
online_q2
target_q1
target_q2
actor_optimizer
q_optimizer
alpha/log_alpha
alpha_optimizer
normalizers
latent_mapper config
basis path + SHA256
resolved config
training step
environment step
replay metadata
RNG states
git commit
feature schema version
```

加载时验证：

- z_dim；
- flow_dim；
- mode数量；
- Q input profile；
- feature schema；
- basis hash；
- preview type；
- BFM model path/checkpoint identifier。

提供：

```text
--resume checkpoint.pt
--weights-only checkpoint.pt
```

---

# 18. Logging

至少支持：

```text
TensorBoard
metrics.jsonl
summary.csv
resolved_config.yaml
run_metadata.json
```

训练日志：

```text
critic/q1_loss
critic/q2_loss
critic/q1_mean
critic/q2_mean
critic/target_mean
critic/disagreement_mean
critic/disagreement_p95
critic/td_abs_mean
actor/loss
actor/entropy
actor/flow_norm
actor/flow_delta_norm
alpha/value
reward/total
reward/<component>
mode/<name>_ratio
rollout/fall_rate
rollout/contact_loss_rate
rollout/retention
rollout/board_progress
rollout/heading_progress
```

不得把每一步的大数组直接写JSON。

---

# 19. Q 测试方案

## 19.1 单元测试

### 网络shape

- 每个input profile均能前向；
- Q输出 `[B,1]`；
- Q1/Q2参数地址不同；
- 最后一层无激活限制；
- no NaN/Inf。

### Target测试

人工小batch验证：

- terminated时不bootstrap；
- truncated且配置允许时仍bootstrap；
- min、mean、mean_minus_std结果正确；
- entropy term开关正确；
- target无gradient。

### BFM冻结

完整一次Q和Policy backward后：

```python
assert all(p.grad is None for p in bfm.parameters())
```

### Adapter一致性

batch preview adapter与现有live adapter对同一29D action输出接近。

## 19.2 Q ranking评测

脚本：

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_q.py \
  --config 03_latent_flow/configs/eval/q_ranking.yaml \
  --checkpoint <path>
```

对每个anchor：

1. 有N个候选flow；
2. 真实执行得到return `G_j`；
3. Q预测 `Q_j`；
4. 计算：

```text
Spearman correlation
Kendall tau
Top-1 regret
Top-3 hit rate
NDCG
failure-last ranking rate
Q1/Q2 disagreement
```

必须分：

```text
push
mount
steer
dismount
recover
overall
```

随机Q/随机排序作为baseline。

## 19.3 OOD与不确定性

测试：

- flow幅度超出训练分布；
- z来自不同prototype区域；
- 摩擦/质量变化；
- 新目标heading；
- 接触异常状态。

观察 `|Q1-Q2|` 是否增大。不要声称Twin Q差异就是可靠不确定性，只把它作为诊断量。

---

# 20. Flow Policy 测试方案

## 20.1 Baselines

至少比较：

1. Stage 01 fixed/phase heuristic；
2. Stage 02 feedback heuristic；
3. zero flow；
4. random local flow；
5. BC-only flow；
6. SAC flow + Q full preview；
7. SAC flow without action preview；
8. SAC flow with minimal Q input。

## 20.2 指标

完整任务：

```text
success rate
time to goal
goal distance reduction
fall rate
episode return
```

滑板交互：

```text
foot-board contact retention
legal contact ratio
robot-board distance
board forward progress
board speed gated by retention
board kicked-away rate
```

模式：

```text
push有效推进率
mount成功率
steer heading改善率
dismount成功率
recover成功率
transition耗时
```

latent：

```text
flow norm
flow temporal difference
z cosine change
z path length
mode-conditioned flow distribution
```

## 20.3 Deterministic evaluation

评测使用：

```text
Flow Policy mean action
BFM mean action
固定seed集合
关闭训练探索
记录视频和JSON summary
```

命令：

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/eval/rollout.yaml \
  --checkpoint <path> \
  --episodes 100 \
  --video-dir /63data1/hwh_data/Skate-bfm/runs/latent_flow/eval_videos
```

---

# 21. 消融系统

实现 `run_ablation.py`，支持通过配置矩阵运行：

```text
q.input_profile:
  minimal
  candidate
  preview
  full_preview

q.target.aggregation:
  min
  mean
  mean_minus_std

q.target.type:
  finite_horizon_return
  semi_mdp_td
  sac_td

q.loss.type:
  huber
  mse

latent.update_type:
  tangent_residual
  euclidean_residual

q.preview.type:
  none
  action_23d
  lower_body_12d

latent.flow_dim:
  8
  16
  32

control.macro_steps:
  2
  5
  10
```

每个run生成唯一目录，保存config和aggregate summary。不要在脚本中手写多个实验分支；通过config override生成。

---

# 22. 初始超参数

默认：

```yaml
latent:
  flow_dim: 16
  step_size: 0.25
  update_type: tangent_residual
  basis_trainable: false

control:
  macro_steps: 5

q:
  input_profile: full_preview
  target:
    type: sac_td
    aggregation: min
    gamma_macro: 0.99
  loss:
    type: huber
  optimizer:
    lr: 3.0e-4
    weight_decay: 1.0e-5
    grad_clip: 10.0
  target_tau: 0.005

policy:
  frame_stack: 5
  optimizer_lr: 3.0e-4

sac:
  batch_size: 512
  replay_capacity: 2000000
  random_steps: 20000
  update_after: 5000
  initial_alpha: 0.1
```

所有值都必须可配置，不可散落在代码中。

---

# 23. 训练稳定性与调试保护

实现以下检查：

1. 输入finite检查；
2. reward finite检查；
3. latent norm检查；
4. action范围与clip比例；
5. Q target范围日志；
6. gradient norm日志；
7. BFM无grad断言；
8. target Q无grad断言；
9. replay字段shape断言；
10. feature schema version；
11. checkpoint/config兼容性检查；
12. 训练出现NaN时保存最近batch和minimal reproduction，不要只退出。

支持debug config：

```yaml
debug:
  anomaly_detection: false
  assert_every_n_steps: 1000
  dump_bad_batch: true
  max_abs_q_warning: 10000
```

---

# 24. 测试与验收条件

Codex完成后必须实际运行测试，不能只生成文件。

## 24.1 静态与单元测试

```bash
source activate.sh
pytest -q 03_latent_flow/tests
```

## 24.2 一次性 Smoke test

允许执行 smoke test，但**不得为了 smoke test 在仓库中保存专用脚本、模块或测试代码**。

执行方式优先采用：

- 终端中的一次性 Python heredoc；
- `python -c`；
- 临时生成到 `/tmp/skate_bfm_latent_flow_smoke_*` 的脚本；
- 复用正式模块和正式单元测试完成最小链路检查。

临时脚本在测试结束后必须删除。不要新增或提交：

```text
03_latent_flow/scripts/smoke_test.py
smoke_test.py
smoke_utils.py
smoke_logs/
smoke_results/
```

Smoke test应覆盖：

- 环境reset；
- 构造Flow actor obs和Critic state；
- policy sample；
- latent map；
- BFM action preview；
- Q1/Q2 forward；
- macro env step；
- replay add/sample；
- 一次critic update；
- 一次policy update；
- target soft update；
- 内存中的checkpoint state_dict组装与恢复检查；
- 验证BFM无gradient。

Smoke test的输出只打印到当前终端，最多给出简洁的PASS/FAIL和关键shape。不得保存：

- smoke日志；
- smoke JSON/CSV；
- TensorBoard事件；
- smoke checkpoint；
- smoke视频；
- smoke dataset；
- smoke运行目录；
- 临时调试batch。

测试结束后清理所有 `/tmp/skate_bfm_latent_flow_smoke_*` 临时文件。正式训练、正式评测和单元测试的日志机制不受此限制。

可采用类似下面的一次性执行方式，但应根据最终接口调整：

```bash
CUDA_VISIBLE_DEVICES=6 python - <<'PY'
# 只导入正式Stage 03模块并执行最小链路检查。
# 不在仓库或data root写入任何文件。
# 最后只print简洁PASS/FAIL。
PY
```

## 24.3 Snapshot test

状态恢复后同一action rollout应在允许误差内一致。

## 24.4 Offline Q overfit test

构造小dataset，训练到明显降低loss并正确排序候选，验证训练链路没有断。

## 24.5 Stage 01 回归

至少运行一次Stage 01原命令，确认新增代码没有改变已有结果路径和导入行为。

---

# 25. README 必须包含

`03_latent_flow/README.md` 至少说明：

1. 架构图；
2. 哪些模块冻结、哪些模块训练；
3. Flow Policy输入输出；
4. Q1/Q2输入输出；
5. Q input profile；
6. Q target/aggregation/loss切换方法；
7. 数据采集；
8. offline Q训练；
9. BC warm start；
10. online SAC训练；
11. Q ranking评测；
12. rollout评测；
13. checkpoint恢复；
14. 数据盘和系统盘约束；
15. 当前已知限制；
16. 常用命令。

---

# 26. 禁止事项

1. 不要只写单文件大脚本；
2. 不要把Q target写进Q网络forward；
3. 不要把feature拼接散落在多个trainer；
4. 不要硬编码输入维度而不做schema验证；
5. 不要共享Q1和Q2的可训练encoder；
6. 不要微调BFM Actor；
7. 不要加载HUSKY Actor参与控制；
8. 不要把reward高等同于滑板任务成功；
9. 不要只报告board speed而忽略人板分离；
10. 不要把大量dataset/checkpoint写入 `/home` 系统盘；
11. 不要添加与本任务无关的大型依赖；
12. 不要删除或重构Stage 01/02，除非是极小且兼容的修复；
13. 不要提交模型权重、视频和大型dataset到Git；
14. 不要用伪造全零特征填补不存在的状态后声称完整实现；
15. 不要跳过测试；
16. 不要将smoke专用脚本、日志、checkpoint、视频、dataset或运行目录保存到仓库、系统盘或数据盘；smoke只能临时执行并在结束后清理。

---

# 27. 实现顺序

严格按以下顺序推进，每完成一项再继续：

1. 审计并写 `IMPLEMENTATION_NOTES.md`；
2. 建立Stage 03目录、配置和schema；
3. 实现feature builders；
4. 实现batch frozen BFM和batch action adapter；
5. 实现latent mapper与basis工具；
6. 实现macro env；
7. 实现Q input profiles；
8. 实现Twin Q；
9. 实现aggregator、target strategy和loss registry；
10. 实现replay；
11. 实现snapshot和branch collector；
12. 实现offline Q trainer与ranking evaluator；
13. 实现Flow Policy和BC；
14. 实现online SAC；
15. 实现rollout evaluator和ablation；
16. 补齐测试与README；
17. 运行一次性smoke、正式unit和regression测试；
18. 清理所有smoke临时文件，确认仓库和数据目录未残留smoke产物；
19. 输出最终报告。

---

# 28. 最终汇报格式

完成后向我汇报：

## A. 审计结果

- 实际复用路径；
- 与Prompt假设不同之处；
- 采取的适配方案。

## B. 新增/修改文件

按目录列出，每个文件一句作用。

## C. 关键架构

- Flow Policy；
- Latent Mapper；
- Twin Q；
- Q calculation registry；
- Macro env；
- Replay；
- Trainer；
- Evaluator。

## D. Q可调选项

明确列出：

```text
input_profile
target.type
target.aggregation
loss.type
preview.type
state_profile
flow_dim
macro_steps
```

## E. 已运行命令

列出实际运行的pytest、一次性smoke、短训练和评测命令。Smoke只需在最终文字报告中概括命令和PASS/FAIL，不得引用或生成持久化smoke日志文件。

## F. 测试结果

- passed/failed；
- 一次性smoke PASS/FAIL（仅在最终文字报告中概括，不引用持久化日志）；
- shape；
- snapshot误差；
- BFM gradient检查；
- 小数据overfit；
- Q ranking初步结果；
- Stage 01回归。

## G. 尚未完成或风险

必须真实说明，不要把未验证功能写成完成。

## H. 下一步运行命令

给出可直接复制的：

1. branch采集；
2. offline Q训练；
3. BC训练；
4. online SAC；
5. Q评测；
6. Flow评测。

---

# 29. 首个可交付里程碑

本次首个完整里程碑至少达到：

```text
Stage 03工程目录完成
配置系统完成
特征构造完成
Latent Mapper完成
Batch BFM preview完成
Twin Q与可切换Q计算方式完成
Replay完成
Macro env完成
Branch collector可运行
Offline Q可训练
Q ranking可评测
Flow Policy + SAC训练链路可通过一次性smoke
不保存任何smoke专用代码、日志或运行产物
Checkpoint和正式训练日志完成
单元测试通过
Stage 01无回归
```

不要求一次长训练就得到稳定滑板成功率，但必须让整个训练和测试闭环真实可运行、可诊断、可继续调参。
