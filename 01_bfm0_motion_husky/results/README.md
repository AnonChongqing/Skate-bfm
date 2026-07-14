# Stage 01: BFM0 + HUSKY 算法与实验记录

本文档同时说明当前算法流程、设计约束、状态机、代码职责和保留的实验
视频。Stage 01 的目标是直接使用冻结的 BFM0 生成 G1 动作，并在 HUSKY
的 MuJoCo/Warp 滑板环境中执行，不使用 HUSKY 已训练 policy。

## 1. 核心边界

当前系统遵守以下约束：

1. 每一步 29 维机器人动作都由冻结的 BFM0 actor 产生。
2. 不加载 HUSKY 的 PPO/AMP checkpoint。
3. 不直接写入预设关节目标来维持平衡或完成 push/steer。
4. 允许根据 HUSKY 实时状态选择、混合和投影 BFM latent。
5. push 和 steer 使用 reward latent；BFM tracking latent 只用于 transition。
6. HUSKY 只负责机器人、滑板、地面、接触、命令和物理仿真。

因此，反馈控制器控制的是 BFM latent，不是 23 个 HUSKY 关节。最终动作
始终经过：

```text
HUSKY state/contact/command
  -> BFM observation adapter
  -> live latent state machine
  -> frozen BFM0 actor
  -> 29D BFM action
  -> joint-name action adapter
  -> 23D HUSKY JointPositionAction
  -> MuJoCo/Warp simulation
```

## 2. 单步算法流程

当前是 **frozen-policy inference baseline**，不是在线强化学习。rollout 期间
BFM0 参数不更新、没有 optimizer、没有 backward。HUSKY reward 有三个用途：

- 记录当前动作在 HUSKY 任务定义下的质量；
- 离线搜索/比较不同 BFM latent；
- 给 BFM `reward_wr_inference()` 标注收集到的状态样本。

正常实时控制不会把 reward 数值直接转换成关节动作；状态机只读取物理状态
并选择 BFM latent，29D action 仍由 actor 推理得到。

一次控制步按以下顺序执行：

1. `Bfm0Husky23Env._create_observation()` 从 HUSKY 读取关节位置、关节速度、
   projected gravity、根角速度和历史动作。
2. `_joint_tensor_bfm_order()` 按 BFM0 的 29DoF 关节顺序重排状态；HUSKY
   不存在的六个 wrist DoF 用默认值或零补齐。
3. `PhaseControl._select()` 读取 phase、板速、板朝向、脚板接触、脚地接触、
   根高度和机器人-滑板距离，选择当前 latent。
4. `BfmPolicy.act()` 把 observation 和 latent 输入冻结的 BFM0 actor，得到
   29D action。
5. `Bfm0ToHusky23ActionAdapter.map_action()` 按关节名抽取 23 个共享动作，
   进行 reference、scale 和 gain 变换。
6. `Bfm0Husky23Env.step()` 把 23D 动作送入 HUSKY，推进 MuJoCo/Warp。
7. `_make_info()` 和 `compute_scores()` 记录板速、距离、接触、phase reward、
   transition goal 和控制器状态。
8. 下一步重新读取状态并闭环选择 latent，不复用预先生成的整段动作。

BFM observation 的主要尺寸为：

| 字段 | 内容 | 尺寸 |
| --- | --- | --- |
| `state` | 29 joint pos + 29 joint vel + 3 gravity + 3 angular velocity | 64 |
| `history_actor` | 4 帧 action、角速度、关节位置/速度和 gravity | 372 |
| `last_action` | 上一步 BFM action | 29 |
| `privileged_state` | 保留的 BFM 接口占位 | 463 |

## 3. 动作适配设计

BFM0 使用 G1-29DoF，HUSKY 当前模型使用 G1-23DoF。共享部分包含腿、腰、
肩和肘；被丢弃的是左右手腕各三个 DoF：roll、pitch、yaw。

默认 `reference` 映射为：

```text
delta = bfm_action
        * BFM_ACTION_SCALE
        * ACTION_RESCALE(5.0)
        * action_gain(1.25)

reference = bfm_default
            + reference_blend * (husky_default - bfm_default)

target_joint_pos = reference + delta
husky_raw_action = (target_joint_pos - husky_default) / husky_action_scale
```

当前 `reference_blend=0.0`，因此 reference 完全使用 BFM0 默认姿态。适配器
只做坐标和尺度转换，不生成额外稳定动作。

## 4. 实时状态机

控制器有外层任务状态机和 push 内层反馈状态机。

```text
外层: push -> push2steer -> steer -> steer2push -> phase wrap -> push
内层: push_hold <-> push_drive
```

### 4.1 Push Hold

`push_hold_z` 默认使用 BFM reward `move-ego-0-0[0]`。它负责保持相对稳定的
零速姿态。进入 push 后至少 hold 15 步，避免一开始就注入强 locomotion。

### 4.2 Push Drive

drive 来源为 `/63data1/hwh_data/Skate-bfm/prompts/push_back.npy`，即
`move-ego-180-0.3[2]`。实际 drive latent 不是完整使用该 prompt，而是：

```text
push_drive_z = project(0.7 * push_hold_z + 0.3 * push_source_z)
```

只有同时满足以下条件才从 hold 进入 drive：

- 右脚接触滑板；
- 左脚接触地面；
- 机器人-滑板 XY 距离小于 0.35 m；
- 当前 hold 已达到 15 步；
- 板速低于 0.4 m/s。

drive 最多持续 8 步；距离超过 0.45 m 或板速超过 0.55 m/s 会提前回到
hold。`push_level` 以每步 `+0.25` 进入、`-0.4` 退出，减少 latent 突变。

### 4.3 Push 到 Steer

transition 不是固定时间强制触发。正常触发要求：

- phase 大于等于 0.12；
- 板速大于等于 0.2 m/s；
- 机器人-滑板距离小于等于 0.25 m；
- 至少一只脚接触滑板；
- 根高度不低于 0.5 m。

`transition_start=0.35` 的 fallback 仍使用同一组速度、距离、接触和高度
门槛，时间不能绕过物理条件。

触发后，`_make_track()` 从当前 23DoF 姿态到 HUSKY `steer_init_pos` 构造
18 步 smoothstep 关节轨迹。轨迹被映射成 BFM observation，然后调用 BFM0
原生 `tracking_inference()` 得到逐帧 tracking latent。

transition 每步 latent 为：

```text
base_z  = project((1-alpha) * current_push_z + alpha * steer_reward_z)
weight  = transition_mix * sin(pi * alpha)
final_z = project((1-weight) * base_z + weight * tracking_z[step])
```

tracking 权重在 transition 中间最大，在首尾为零，减少 reward latent 和
tracking latent 的切换冲击。

当前 tracking 是 **joint-space approximation**：HUSKY 的 `steer_init_pos`
先形成 23DoF 关节轨迹，再适配成 BFM observation。它还没有把 HUSKY 的
key-body Bezier/Slerp goal 经过 IK 精确转换成 BFM full-body goal。

当前显式 18 步状态机实际使用 `transition_scale`、`transition_mix` 和
`transition_steps`。`transition_enter`、`transition_blend` 只由保留的通用
`_track_prompt()` 使用，不影响当前 `_select()` 主路径。

### 4.4 Steer

steer 根据实时 heading error 选择正向或负向 `rotate-z` reward latent：

```text
turn_sign = sign(target_heading - board_heading)
turn_mix  = min(abs(heading_error) / 0.5, 1.0)
            * configured_turn_mix
            * stability
steer_z   = project((1-turn_mix) * base_steer_z + turn_mix * signed_rotate_z)
```

`stability` 根据根高度得到。phase 控制默认 `turn_mix=0.05`，仅提供弱转向
提示；直接板上测试使用 `move-ego-0-0[0]` 作为保持基座，并混入 10% 的
`rotate-z[9]`。目标 `0.4 rad` 在第 47-48 步达到，当时仍有脚板接触；但
120 步测试在此后持续脱板，因此这是短时 steer baseline，不是持续控制。

`SteerControl` 还提供实时 BFM latent 反馈。它把板中心相对位置、相对速度、
projected gravity 和脚到板面 marker 的误差组合成机器人局部二维平衡误差，
再从前/后/左/右四个 BFM reward latent 中选择方向并逐渐混合。最佳配置只启用
脚板 marker 项，最大混合 5%。它把首次完全脱板从第 35 步推迟到第 47 步，
并把 120 步最终距离从 `0.914 m` 降到 `0.574 m`；但第 58 步后仍持续脱板。

### 4.5 Steer 到 Push

进入 HUSKY `steer2push` phase 后，控制器构造到默认姿态的 tracking
trajectory，并用同样的正弦 envelope 混合 tracking latent。HUSKY phase
回绕时，缓存和状态机重置为 push。

## 5. Reward 与评价设计

`compute_scores()` 复用 HUSKY 论文任务项，但额外加入 board retention，防止
搜索器把“踢走滑板”误判成成功。

### Push

```text
push_quality = (
    3.0 * speed_tracking
  + 1.0 * yaw_alignment
  + 3.0 * left_foot_air_time
  + 0.5 * ground_ankle_parallel
) / 7.5

board_proximity = exp(-(board_distance / 0.45)^2)
board_retention = board_proximity * (0.25 + 0.75 * board_contact)
push_task = push_quality * board_retention * upright
```

这意味着正板速只有在机器人仍靠近滑板、保持接触且直立时才有较高分。

### Steer

`steer_quality` 组合脚板接触数量、steer 姿态、双脚距离、heading tracking
和 tilt guide，权重分别为 `3.0, 1.5, 1.0, 5.0, 4.0`。

### Transition

`transition_goal()` 只在 transition phase 计算 HUSKY 规划 key-body 的位置和
旋转 tracking，最终为两者均值。它当前用于评价 transition，不直接生成
动作；控制侧 BFM tracking 使用的是上一节所述 `steer_init_pos` 关节轨迹。
因此现版本是 goal-score evaluation 加 joint-space tracking approximation，
不在 push 或 steer 阶段持续使用 goal tracking。

## 6. Prompt 搜索与 BFM Reward Inference

`scripts/search.py` 的流程是：

1. 从官方 reward latent、随机 latent 或两个 latent 的投影混合构造 explorer。
2. 每个 explorer 启动独立 `rollout.py` 子进程，避免 MuJoCo/Warp 同进程 reset
   造成不可重复的假提升。
3. 收集每个时刻的 BFM observation 和 HUSKY task reward。
4. inference reward 使用：

```text
upright * (0.45 * board_contact * proximity + 0.55 * push_task)
```

5. 把所有样本送入 BFM0 `reward_wr_inference()`，生成新的 256D latent。
6. 同时保存 reward-inferred latent 和最佳原始 explorer，二者必须分别做独立
   rollout 验证。

steer-balance inference 使用双脚接触、至少一脚接触、板接近度、官方 steer
姿态、脚间距和 heading error 标注 600 个状态。直接使用 inferred latent 在
第 26 步跌倒；1%-2% residual 也未提高接触，说明一次 backward-map inference
不能解决 skateboard domain gap。

`scripts/adapt.py` 进一步用 CEM 在冻结 BFM latent manifold 上做 few-shot
adaptation。第一轮平均接触从 86.7% 提到 91.7%，但 120 步盲测没有提高总接触。
加入脚板 marker 闭环后，连续接触从 34 步提高到 46 步；第二轮局部 CEM 没有
超过该基准。随后独立遍历四方向 locomotion reward 的 10 个 recovery latent，
索引 0 仍是最佳；索引 3、8 只能持平 46 步且综合分更低。当前 frozen
actor/latent 局部搜索已接近上限。

## 7. 主要设计决策

### Prompt-level closed loop

BFM0 原 actor 保持冻结，反馈只发生在 latent 层。这样可以验证 BFM0 motion
prior 在 HUSKY 接触域中的能力，同时避免用手写关节动作掩盖兼容性问题。

### Physical safety gates

transition 同时要求速度、接触、距离和高度。仅到达某个 phase 或板速瞬间
升高不能触发 transition，防止 kick-away 轨迹被当成合理控制序列。

### Reward anti-hacking

搜索 objective 不单看 board velocity，而使用 `board_retention`、接触率、
机器人-滑板距离和 upright。`03_kickaway_failure.mp4` 就是加入这些约束的
直接原因。

### Independent-process evaluation

每个 search/tune candidate 在全新进程中创建 MuJoCo/Warp 环境。早期同进程
反复 reset 会产生不可重复的状态残留，使参数看起来虚假提升。

### Explicit 29DoF/23DoF ownership

关节映射完全按名称建立，六个 wrist DoF 显式记录为 dropped。代码拒绝
HUSKY 中 BFM 不认识的关节，避免静默错位。

## 8. 代码结构与函数职责

以下表格覆盖 Stage 01 自己维护的集成代码。`bfm0/` 和 `husky_sim/` 是带
upstream 记录的本地上游代码，边界在后文单独说明。

### 8.1 `skate_bfm01/control.py`

| 类/函数 | 作用 |
| --- | --- |
| `_wrap()` | 把 heading error 包装到 `[-pi, pi]`。 |
| `FixedControl.__init__()` | 保存 BFM policy，建立固定 latent baseline 的状态字段。 |
| `FixedControl.__call__()` | 不切换 latent，直接调用 BFM actor。 |
| `SteerControl.__init__()` | 加载零速保持、正/负旋转和四方向跟随 reward latent，保存航向与位置反馈参数。 |
| `SteerControl.__call__()` | 读取实时板航向、根高度、脚板接触和相对位置，闭环混合 BFM latent，最后调用一次冻结 actor 生成 29D action。 |
| `PhaseControl.__init__()` | 加载 push/hold/steer/rotate latent，解析 transition 和触发参数，初始化双层状态机及 tracking 缓存。 |
| `_steer_prompt()` | 根据 heading error 符号、大小和稳定度混合 signed rotate latent。 |
| `_smoothstep()` | 提供 `[0,1]` 平滑插值曲线。 |
| `_make_track()` | 从当前姿态到目标姿态生成平滑轨迹，并调用 BFM tracking inference。 |
| `_feedback_push()` | 根据脚板/脚地接触、距离、板速和持续步数切换 `push_hold`/`push_drive`，平滑更新 push latent。 |
| `_track_prompt()` | 通用的 phase-based reward/tracking 混合辅助函数；当前主状态机 transition 使用显式 18 步 `_make_track()` 路径，此函数保留供其他 phase 方案使用。 |
| `_select()` | 读取实时 HUSKY 状态，执行外层状态机，返回本步 latent、状态、phase、blend 和 heading error。 |
| `PhaseControl.__call__()` | 调用 `_select()`、更新日志状态，然后让 BFM actor 解码动作。 |
| `make_control()` | 根据 `fixed`、`phase` 或 `steer` CLI 配置构建控制器。 |

### 8.2 `skate_bfm01/policy.py`

| 类/函数 | 作用 |
| --- | --- |
| `load_data()` | 用 pickle 读取 prompt；失败时回退 joblib。 |
| `load_zs()` | 从 `.npy` 或 reward prompt 字典读取并统一成 `[N, 256]` tensor。 |
| `load_z()` | 按 key/index 选择单个 latent，并检查索引。 |
| `BfmPolicy.__init__()` | 验证使用的是项目内 vendored BFM0，加载 checkpoint 和默认 latent。 |
| `_batch()` | 把单步 observation 转到 BFM device 并增加 batch 维。 |
| `act()` | 调用冻结 actor，从 observation 和 latent 生成 29D 动作。 |
| `infer_goal()` | 调用 BFM0 goal inference，得到目标状态 latent。当前主流程未使用。 |
| `infer_tracking()` | 对批量目标轨迹调用 BFM0 tracking inference。transition 使用此函数。 |
| `infer_reward()` | 对 observation/reward 样本调用普通或加权 BFM reward inference。搜索脚本使用。 |
| `project()` | 把 latent 投影回 BFM0 规定的 latent manifold/norm。 |
| `__call__()` | `act()` 的默认 latent 简写。 |

### 8.3 `skate_bfm01/envs/env.py`

| 类/函数 | 作用 |
| --- | --- |
| `Bfm0Husky23EnvCfg` | 集中定义 task、device、action/observation mapping、命令、随机化、history 和渲染配置。 |
| `Bfm0Husky23Env.__init__()` | 加载项目内 HUSKY task，应用 play/randomization/command 配置，创建仿真、动作适配器、关节映射和历史 buffer。 |
| `reset()` | 重置 HUSKY，根据 initial mode 选择 push/steer 初态，清空 BFM history 并返回首个 observation。 |
| `_reset_steer_state()` | 使用官方 `steer_start_pose_b` 与 `steer_init_pos`，按板面 marker 对齐双脚，并给机器人和板相同初速度。 |
| `step()` | 校验 29D action、映射成 23D、推进 HUSKY、更新 history，并返回结构化 info。 |
| `close()` | 关闭 HUSKY 仿真和渲染资源。 |
| `render()` | 返回 HUSKY RGB frame。 |
| `observation` | 暴露当前 BFM observation。 |
| `goal_observation()` | 把单个 HUSKY joint pose 转成 BFM goal observation。 |
| `tracking_observation()` | 把 HUSKY joint trajectory 转成批量 BFM state/history，用于 tracking inference。 |
| `mapping_report` | 返回共享和被丢弃关节的报告。 |
| `set_calibration()` | 运行时更新 reference blend 和 action gain。 |
| `_as_action_tensor()` | 把输入规范为 device 上的 `[1,29]` float tensor。 |
| `_joint_tensor_bfm_order()` | 按关节名把 HUSKY tensor 重排/补齐为 BFM 29DoF 顺序。 |
| `_joint_reference()` | 根据 observation mapping 计算 BFM joint position reference。 |
| `_roll_history()` | 将最新值推入固定长度 history buffer。 |
| `_create_observation()` | 构造 BFM `state/history_actor/last_action/privileged_state`。 |
| `_make_info()` | 汇总位置、板速、heading、距离、接触、命令、phase 和实际 HUSKY action。 |

### 8.4 `skate_bfm01/adapters/actions.py`

| 类/函数 | 作用 |
| --- | --- |
| `ActionMappingReport` | 保存 BFM/HUSKY/共享/丢弃关节名及映射模式。 |
| `Bfm0ToHusky23ActionAdapter.__init__()` | 按关节名建立 29D 到 23D 索引，缓存默认姿态和动作尺度，拒绝未知 HUSKY 关节。 |
| `report` | 生成只读映射报告。 |
| `map_action()` | 执行 shared joint 抽取、reference/scale/gain 变换和可选 clipping。 |
| `set_calibration()` | 更新 reference blend 和 action gain。 |

### 8.5 `skate_bfm01/score/`

| 文件/函数 | 作用 |
| --- | --- |
| `reward.py::ScoreCfg` | 定义 push/steer 距离核和 goal 配置。 |
| `reward.py::compute_scores()` | 计算 HUSKY push/steer 项、board retention、upright、phase mask 和总 active score。 |
| `goal.py::GoalCfg` | 定义 transition body position/rotation 的标准差。 |
| `goal.py::transition_goal()` | 只在 transition 目标上计算 key-body 位置/旋转 tracking score。 |

### 8.6 `skate_bfm01/constants.py`

该文件没有函数，定义 BFM 29DoF 顺序、六个 wrist 名称、每关节 action scale、
BFM 默认姿态和全局 `ACTION_RESCALE=5.0`。observation 和 action adapter 必须
共同使用这套顺序，否则动作会落到错误关节。

### 8.7 `skate_bfm01/viewer.py`

| 类/函数 | 作用 |
| --- | --- |
| `ViewerEnv.__init__()` | 把 BFM wrapper 封装成 MJLab viewer 协议。 |
| `cfg` / `unwrapped` | 向 viewer 暴露底层 HUSKY 配置和环境。 |
| `get_observations()` | 返回当前 BFM observation。 |
| `reset()` / `step()` / `close()` | 代理环境生命周期。 |
| `run_viser()` | 设置端口，以仿真 step rate 启动 `ViserPlayViewer`。 |

### 8.8 `scripts/rollout.py`

| 函数 | 作用 |
| --- | --- |
| `_resolve_output()` | 将相对输出路径解析到项目根目录。 |
| `write_video()` | 使用 MoviePy 把 RGB frame 序列编码成 MP4。 |
| `main()` | 解析全部 CLI；加载 BFM、HUSKY 和控制器；执行 rollout；逐步评分/记录；选择 JSON、MP4 或 Viser 输出。隐藏的 `--samples` 会保存 reward inference 所需 observation。 |

### 8.9 `scripts/search.py`

| 函数 | 作用 |
| --- | --- |
| `_mean()` | 对可能为空的数值列表求均值。 |
| `_summarize()` | 计算跌倒、接触率、retention、保板条件下速度、距离和防 kick-away objective。 |
| `_collect_trial()` | 为单个 candidate 建临时 latent，启动独立 rollout 子进程，读取 JSON 和压缩 observation 样本。 |
| `main()` | 构造 official/random/blended explorer，汇总样本，调用 BFM reward inference，并分别保存 inferred 和 best-explorer latent。 |

### 8.10 `scripts/adapt.py`

| 函数 | 作用 |
| --- | --- |
| `_mean()` | 对可能为空的数值列表求均值。 |
| `_score()` | 联合连续接触、双脚接触、retention、steer pose、heading、存活和距离评价候选。 |
| `_rollout()` | 为单个 latent 启动隔离的 HUSKY steer rollout，并读取结构化结果。 |
| `main()` | 围绕稳定 latent 做 CEM 采样、manifold 投影和精英更新，保存最佳 frozen-BFM latent 与报告。 |

### 8.11 `scripts/tune.py`

| 函数 | 作用 |
| --- | --- |
| `_values()` | 把逗号分隔 CLI 网格转换成目标类型列表。 |
| `_summarize()` | 以跌倒或距离超过 0.6 m 作为有效轨迹终点，综合存活、transition、接触、位移和 retention。 |
| `_run_trial()` | 每组超参数启动独立 rollout，避免同进程仿真 reset 污染比较。 |
| `main()` | 构造 transition、turn、action gain 和 push hold/drive 参数笛卡尔积，选择最高 objective。 |

### 8.12 其他脚本与配置

| 文件/函数 | 作用 |
| --- | --- |
| `scripts/smoke.py::_resolve_output()` | 解析 smoke 输出路径。 |
| `scripts/smoke.py::_jsonable()` | 把 tensor/array/dict 递归转换为 JSON 类型。 |
| `scripts/smoke.py::main()` | 用零或小随机 29D action 验证环境、维度和 adapter，不加载 BFM checkpoint。 |
| `scripts/sweep.py::main()` | 隔离遍历 push、phase turn 或 direct-steer 的基座/旋转/recovery latent，联合统计连续接触、总接触、距离、heading、位移和存活。 |
| `configs/default.yaml` | 当前单环境、命令、adapter、状态机和输出目录默认值。 |
| `configs/husky_score.yaml` | HUSKY push/steer/transition 评分权重的记录。 |
| 各级 `__init__.py` | 导出 env、adapter 和 score 公共接口，不包含算法逻辑。 |

## 9. 上游代码边界

### `bfm0/bfm_zero_inference_code/`

这是项目内 vendored BFM0 推理核心：

- `base.py`、`base_model.py`：配置、模型序列化和 checkpoint 加载。
- `fb/model.py`：FB actor、latent projection、`act()`、`reward_inference()`、
  `goal_inference()`、`tracking_inference()`。
- `fb_cpr_aux/model.py`：带 CPR/auxiliary reward 的实际 checkpoint 类型。
- `nn_models.py`、`nn_filters.py`、`normalizers.py`：网络、输入过滤和归一化。
- `inference/rewards.py`、`utils.py`：官方 locomotion/rotate/crouch 等 reward
  定义和 tolerance 工具。

Stage 01 不修改 BFM actor 输出，也不在这些网络层中加入 HUSKY 专用动作。

### `husky_sim/`

这是项目内 HUSKY/MJLab 仿真副本，提供 G1-23DoF、滑板 MJCF、接触 sensor、
phase、命令、terrain、HUSKY reward 和 Viser viewer。Stage 01 通过 wrapper
调用它，不使用其训练 policy。

## 10. 当前证据视频

| 视频 | 说明 | 实测结果 |
| --- | --- | --- |
| `01_best_transition.mp4` | 当前最佳早期 transition 尝试。能进入 push-to-steer 并保持直立，但脚板接触间歇、人与板距离持续增加。 | 180 步；最低根高度 0.706 m；平均板速 +0.193 m/s；板接触率 24.4%；最终距离 0.680 m。 |
| `02_safe_reactive.mp4` | contact-aware hold/drive。约前 100 步稳定靠近滑板，但没有持续正向推进，正速度门正确阻止 transition。 | 190 步；最低根高度 0.665 m；平均板速 -0.129 m/s；板接触率 38.4%；最终距离 0.927 m。 |
| `03_kickaway_failure.mp4` | 说明为什么不能只看板速：reverse prompt 把板踢走、失去接触，机器人随后跌倒。 | 190 步；最低根高度 0.111 m；平均板速 +0.760 m/s；板接触率 2.6%。 |
| `04_direct_steer.mp4` | frozen-BFM CEM latent + signed rotate + 实时脚板 marker latent 反馈。视频完整保留成功窗口和后续脱板，避免只展示截断片段。 | 120 步；首次完全脱板从基线第 35 步推迟到第 47 步；最低根高 0.724 m；最终距离从 0.914 m 降到 0.574 m；第 58 步后仍持续脱板。 |

## 11. 当前结论

系统已经实现 BFM0-only 的 observation adapter、action adapter、实时 latent
反馈、tracking transition、signed steer、reward inference、CEM latent
adaptation、独立搜索和两种可视化。当前未解决
的是同时满足：

1. 正向滑板速度；
2. 持续脚板接触；
3. 有界机器人-滑板距离；
4. 稳定 transition 和 heading steer。

正板速但无接触属于 kick-away failure，不是滑板成功。下一步需要更丰富的
当前 frozen actor 的 observation 不直接包含板和脚板接触，外层 scheduler
只能切换已有 locomotion latent，无法形成真正的接触反射。下一步若要求持续
平衡，需要训练一个读取板/接触状态的 BFM latent scheduler，或在保留 BFM
motion prior 的前提下做 skateboard-domain actor adaptation。

## 12. 运行命令

录制当前 safe reactive 版本：

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
source activate.sh

CUDA_VISIBLE_DEVICES=6 ./run.sh \
  --device cuda --husky-device cuda:0 --mean \
  --control phase --steps 190 \
  --output /63data1/hwh_data/Skate-bfm/runs/default_final.json \
  --video /63data1/hwh_data/Skate-bfm/runs/default_final.mp4
```

Viser：

```bash
CUDA_VISIBLE_DEVICES=6 ./run.sh \
  --device cuda --husky-device cuda:0 --mean \
  --control phase --viewer viser --port 8080 --steps 0
```

`--video` 和 `--viewer viser` 是互斥模式。大体积 JSON、搜索报告和新视频
默认写入 `/63data1/hwh_data/Skate-bfm/runs/`，不再污染源码 `results/`。
