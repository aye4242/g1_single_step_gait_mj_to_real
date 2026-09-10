# Unitree G1 单阶楼梯盲走 PPO：从训练到 Sim2Real 的完整实现路线

## 1. 项目目标

目标是在 Unitree G1 29DoF 人形机器人上完成以下闭环：

```text
MuJoCo/MJLab 训练
    -> Play 与指标评估
    -> 导出 ONNX
    -> unitree_mujoco + G1 控制器验证
    -> Unitree SDK2 真机部署
    -> 单阶楼梯上行和下行
```

第一阶段采用盲走策略：策略不使用深度图、RGB 图像或 height scan，只使用真机可获得的本体感知信息。上楼梯和下楼梯分别训练两个策略，以降低任务难度并便于定位问题。

成功不定义为“某一次成功跨过楼梯”，而定义为在一组随机初始状态、楼梯高度、摩擦和控制噪声下，策略能够稳定完成任务，并在楼梯平台或落地面恢复站立/行走。

## 2. 两个仓库的职责边界

### 2.1 主工程底座：`unitree_rl_mjlab`

仓库：[unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)

保留其现有能力：

- MuJoCo/MJLab 仿真环境；
- Unitree G1 29DoF 机器人模型；
- RSL-RL PPO 训练器；
- G1 Flat/Rough 任务注册、训练和 Play；
- ONNX 导出；
- `unitree_mujoco` 部署仿真；
- 官方 G1 SDK2 C++ 控制器和真机部署链路。

该仓库当前的 G1 任务主要是 Flat/Rough，并不是现成的单阶楼梯任务。因此需要在其任务结构中新增单阶上行和下行任务。

### 2.2 任务设计参考：`G1DWAQ_Lab`

仓库：[liuyufei-nubot/G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab)

参考并迁移以下设计思想：

- 盲走 actor 的本体感知输入；
- 多帧历史观测；
- actor/critic 非对称观测；
- 上楼与下楼地形；
- 地形课程学习；
- 姿态、足端、滑动、碰撞、能耗和动作平滑奖励；
- 摩擦、质量、执行器和延迟随机化；
- 真机部署时观测历史缓存和动作接口的处理方式。

第一阶段不直接迁移其 `model_9999.pt`、`ActorCritic_DWAQ`、`DWAQPPO` 或完整 VAE 训练代码。两个仓库的 checkpoint、runner、观测 ABI 和导出流程不同，不能直接互换。

## 3. 第一阶段算法定义

采用：

```text
标准 PPO
+ MLP Actor/Critic
+ Actor 使用本体感知历史
+ Critic 使用仿真特权信息
+ 无视觉输入
+ 上行和下行独立策略
```

可称为 `Blind Proprioceptive PPO`，即带历史观测的盲走 PPO。

### 3.1 Actor 输入

Actor 只能使用部署时 G1 能直接提供的信息：

| 观测 | 维度 | 部署来源 |
|---|---:|---|
| 机身角速度 | 3 | IMU |
| 重力投影 | 3 | IMU 姿态 |
| 速度命令 | 3 | 遥控器或上层状态机 |
| 关节位置偏差 | 29 | 电机状态 |
| 关节速度 | 29 | 电机状态 |
| 上一步动作 | 29 | 控制器缓存 |
| 步态相位 | 2 或 4 | 控制器内部时钟 |

单帧约 96～100 维。第一版使用最近 5 帧历史，展平后输入 MLP：

```text
[obs(t-4), obs(t-3), obs(t-2), obs(t-1), obs(t)]
    -> Actor MLP
    -> 29 个关节位置动作
```

训练和部署必须使用相同的历史长度、时间顺序、归一化、动作缩放和裁剪规则。

### 3.2 Critic 输入

Critic 可以使用训练期间仿真才有的特权信息：

- base 线速度和高度；
- terrain height scan；
- 楼梯相对位置和高度；
- 足端位置、速度、接触状态和接触力；
- 地面摩擦系数；
- 其他用于价值估计的仿真状态。

这些信息不能进入最终 Actor。这样训练的是盲走策略，而不是依赖仿真地形信息的策略。

## 4. 推荐仓库结构

在 `unitree_rl_mjlab` 中新增任务，不直接破坏原有 G1 Flat/Rough：

```text
src/tasks/single_step/
    __init__.py
    terrains.py
    observations.py
    rewards.py
    terminations.py
    curriculums.py

src/tasks/velocity/config/g1_single_step_up/
    __init__.py
    env_cfgs.py
    rl_cfg.py

src/tasks/velocity/config/g1_single_step_down/
    __init__.py
    env_cfgs.py
    rl_cfg.py

deploy/robots/g1/config/policy/single_step_up/
deploy/robots/g1/config/policy/single_step_down/
```

建议注册两个任务：

```text
Unitree-G1-Single-Step-Up
Unitree-G1-Single-Step-Down
```

## 5. 实现阶段与验收标准

### 阶段 0：固定版本与环境

记录以下版本，后续所有训练和部署都使用同一份记录：

- Ubuntu、Python、CUDA、PyTorch；
- MuJoCo/MJLab、RSL-RL、`unitree_rl_mjlab` commit；
- `unitree_mujoco` commit；
- Unitree SDK2 版本；
- G1 型号、29DoF 关节配置和网络接口。

验收：能够安装仓库并列出 G1 任务，代码仓库保持干净、配置可重复。

### 阶段 1：复现官方 G1 Flat

先不改楼梯任务，跑通：

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
python scripts/play.py Unitree-G1-Flat --checkpoint_file=<checkpoint>
```

验收：

- G1 可以稳定站立和低速行走；
- reward、episode length、跌倒率可记录；
- 能生成 ONNX；
- `unitree_mujoco` 中可以用 `g1_ctrl --network=lo` 运行；
- 真机部署代码的观测、动作和关节映射已确认。

这一阶段的目的是先验证控制链路，而不是追求楼梯效果。

### 阶段 2：建立盲走平地基线

复制 G1 Rough 配置，但删除 Actor 的 `height_scan`，保留 Critic 的地形和接触特权信息。加入 4～5 帧本体感知历史。

任务仍然是平地或轻微粗糙地形。

验收：

- 无 height scan 的 Actor 仍能稳定平地行走；
- Play 和导出后的 ONNX 行为一致；
- 在 `unitree_mujoco` 中没有明显抖动、漂移或启动姿态错误；
- 真机平地低速测试通过后，才进入楼梯训练。

### 阶段 3：实现单阶上行环境

场景结构：

```text
低平台 ------------------┐
                         │  单阶高度 h
                         └---------------- 高平台
```

每个 episode 应覆盖完整过程：

```text
低平台接近 -> 前脚登阶 -> 后脚登阶 -> 高平台恢复稳定
```

第一版建议随机化：

- 台阶高度：从 2～4 cm 开始，逐步增加到目标高度；
- 台阶深度：约 0.28～0.36 m；
- 初始距离和左右偏移；
- 初始 yaw 误差；
- 摩擦系数；
- 接近速度和停止命令；
- 台阶边缘位置的小范围偏差。

上行奖励至少包括：

- 线速度跟踪；
- 身体姿态和角速度稳定；
- 横向漂移惩罚；
- 足端滑动惩罚；
- 非足端碰撞惩罚；
- 动作变化率、关节加速度和能耗惩罚；
- 足端摆动高度；
- 进入高平台后的稳定奖励；
- 跌倒和危险接触终止惩罚。

验收：在仿真随机测试集上统计成功率，而不是只看训练 episode 的平均 reward。

### 阶段 4：实现单阶下行环境

场景结构：

```text
高平台 ------------------┐
                         │  单阶高度 h
                         └---------------- 低平台
```

完整 episode：

```text
高平台接近边缘 -> 前脚下阶 -> 吸收落地冲击 -> 后脚下阶 -> 低平台恢复稳定
```

下行必须单独设计奖励，不能只把上行地形翻转。重点加入：

- 足端落地垂直速度惩罚；
- 足端冲击力或接触冲量惩罚；
- base 向下速度和俯仰角速度惩罚；
- 膝关节缓冲相关奖励；
- 双脚到达低平台后的恢复稳定奖励；
- 边缘附近的危险躯干碰撞终止。

下行课程建议比上行更保守：

```text
2 cm -> 4 cm -> 6 cm -> 8 cm -> 目标高度
```

验收：下行成功率、最大躯干俯仰角、最大接触冲量和跌倒率都要单独记录。

### 阶段 5：域随机化与鲁棒性训练

先在固定参数下得到可行策略，再逐步加入随机化：

| 项目 | 初始范围建议 |
|---|---|
| 静/动摩擦 | 以真实地面测量值为中心，逐步扩展 |
| 控制延迟 | 0～2 个控制周期 |
| 关节位置噪声 | ±0.005～0.015 rad |
| 关节速度噪声 | 按真实状态反馈误差标定 |
| IMU 噪声 | 按真实 IMU 日志标定 |
| Kp/Kd | ±10%～20% |
| 质量和质心 | ±5%～10%，质心 ±1～3 cm |
| 台阶高度 | 目标值附近小范围随机 |
| 初始距离/偏移 | 覆盖真机可控范围 |

不要一开始同时使用所有宽范围随机化，否则难以判断训练失败原因。

### 阶段 6：部署仿真与 ONNX 检查

训练完成后执行：

```text
训练 checkpoint
    -> ONNX 导出
    -> 由官方 G1 控制器加载
    -> unitree_mujoco 中运行
```

重点检查：

- 29 个关节的顺序和电机索引；
- 默认关节角；
- action scale；
- Kp/Kd；
- 控制频率和 decimation；
- IMU 坐标系、重力投影和四元数顺序；
- 速度命令的符号、缩放和限幅；
- 历史观测的新旧顺序；
- ONNX 输入 shape、dtype 和输出范围；
- 策略启动、停止、急停和阻尼模式。

### 阶段 7：真机平地到楼梯

真机必须按风险递增顺序验证：

```text
悬挂状态动作检查
    -> 悬挂状态站立流程
    -> 平地原地站立
    -> 平地低速行走
    -> 平地停止和急停
    -> 低矮软质单阶上行
    -> 低矮软质单阶下行
    -> 目标高度上行
    -> 目标高度下行
    -> 上行后站稳再下行
```

首次真机测试必须使用吊装/保护绳、低速命令、急停人员和足够缓冲空间。没有通过平地和低矮台阶测试时，不得直接测试目标高度楼梯。

## 6. 任务切换方式

纯盲走 Actor 无法可靠判断前方楼梯的类型和位置，因此第一版不要让策略承担“识别上楼还是下楼”的职责。

使用人工遥控器或上层状态机切换：

```text
Flat policy       平地行走
Step-up policy    单阶上行
Step-down policy  单阶下行
```

上行策略和下行策略都应包含楼梯前后的平地阶段，不能只训练楼梯接触瞬间。初始距离、朝向和速度范围应限制在策略可观察和可控制的范围内。

## 7. 训练与部署数据流

### 训练

```text
MuJoCo 状态
    -> Actor 本体观测 + 历史缓存
    -> Actor 动作
    -> 关节位置控制
    -> 环境状态转移
    -> reward / done

MuJoCo 特权状态
    -> Critic 观测
    -> PPO value estimate
```

### 部署

```text
G1 LowState
    -> IMU、关节位置、关节速度
    -> 与命令和上一动作拼接
    -> 更新 5 帧历史
    -> ONNX Actor
    -> 29 个目标关节位置
    -> SDK2 LowCmd
```

部署时没有 Critic、terrain height scan、reward 或仿真特权状态。

## 8. 必须维护的 Policy Contract

建议为每个策略保存一份 YAML/JSON 合同，训练导出和 C++ 部署都读取或校验：

```text
robot: unitree_g1_29dof
task: single_step_up / single_step_down
num_actions: 29
single_obs_dim: 96 or 100
history_length: 5
control_dt: 0.02
action_scale: ...
default_joint_pos: ...
joint_order: ...
motor_index_map: ...
imu_frame: ...
command_order: ...
observation_scales: ...
```

这样可以避免 Python 训练端和 C++ 部署端分别手写观测，减少 sim2real 中最常见的维度、符号和关节映射错误。

## 9. 指标与实验记录

每次训练至少记录：

- 训练配置和 Git commit；
- episode reward 和 episode length；
- 平地行走成功率；
- 上行成功率；
- 下行成功率；
- 跌倒率；
- 最大躯干俯仰/横滚角；
- 最大足端接触冲量；
- 足端滑动距离；
- 任务完成时间；
- 不同台阶高度、摩擦和初始偏移下的结果；
- Isaac/MJLab、部署仿真和真机之间的差异。

测试集必须与训练随机种子、初始状态和地形实例分离。没有成功率和失败分类，不应只依据 reward 判断策略已经具备真机能力。

## 10. 后续可选：完整 DWAQ

当标准盲走 PPO 已经完成以下目标后，再考虑迁移 DWAQ：

- G1 Flat 训练和部署闭环已跑通；
- 上行和下行标准 PPO 均能在 MuJoCo 中稳定完成；
- ONNX 与部署控制器输出一致；
- 真机平地行走稳定；
- 低矮单阶真机测试成功；
- 已经有明确的失败样本和传感器日志。

再增加：

- 历史观测 encoder；
- latent code；
- 速度估计损失；
- 重构损失和 KL 正则；
- DWAQ 专用 runner；
- 对应的 TorchScript/ONNX 导出和部署侧推理。

这样可以将标准 PPO 作为基线，明确比较 DWAQ 是否改善上下阶成功率，而不是一开始同时引入多个难以排查的变量。

## 11. 最终路线总结

```text
1. 固定版本和硬件接口
2. 复现 unitree_rl_mjlab 的 G1 Flat
3. 完成 ONNX、unitree_mujoco 和 g1_ctrl 闭环
4. 做无视觉、无 height scan 的平地盲走 PPO
5. 新增单阶上行任务
6. 新增单阶下行任务
7. 分别训练上行和下行策略
8. 加入真实参数驱动的域随机化
9. 在部署仿真中验证导出策略
10. 真机平地验证
11. 低矮台阶验证
12. 目标高度上行、下行和连续流程验证
13. 根据实验结果决定是否加入完整 DWAQ
```

最终系统的推荐定位是：

> `unitree_rl_mjlab` 负责 MuJoCo 训练、RSL-RL PPO、ONNX 导出和 Unitree G1 部署；`G1DWAQ_Lab` 负责提供盲走楼梯任务的观测、奖励、课程和域随机化参考。第一版实现标准历史观测 PPO，而不是直接移植 DWAQ checkpoint。

## 12. 主要参考

- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [liuyufei-nubot/G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab)
- [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)

文档中的成功率、真机稳定性和目标台阶参数需要通过实际实验测量，不能仅根据仓库 README 或仿真 checkpoint 推断。
