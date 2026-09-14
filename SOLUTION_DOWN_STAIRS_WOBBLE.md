# 下楼梯落地后摇摇晃晃 - 解决方案

## 问题现象
下台阶后落地瞬间躯干左右或前后晃动，需要2-3步才能恢复稳定。

## 根本原因
训练时为了避免"不敢下台阶"而删除了关键稳定性奖励：
```python
del cfg.rewards["soft_landing"]      # 删除落地缓冲奖励
del cfg.rewards["angular_momentum"]  # 删除角动量惩罚
cfg.rewards["body_orientation_l2"].weight = -0.3  # 躯干姿态降到很低（原-1.0）
```

**结果：** 策略学会了"敢下台阶"，但没学会"稳稳落地"。重力势能转化为落地冲击，策略不会主动缓冲和控制角动量。

---

## 快速修复方案（无需重训练）

### 方案 1：降低 Kd 阻尼增益 ⭐ 优先尝试
修改 `deploy_single_step/configs/g1_single_step_down.yaml`：

```yaml
# 原配置（与上楼梯相同）
kds: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# 修改为（髋关节和膝关节降低40%）
kds: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
#     ↑髋关节 yaw/roll/pitch 各 5→3，减少高频振荡
```

**原理：**
- 下台阶落地冲击 → 髋关节高速压缩 → 高 Kd 放大振荡 → 躯干晃动
- 降低 Kd → 允许关节"柔顺吸收"冲击 → 减少反向力矩 → 更稳

**优点：** 立即生效，无需重训练
**缺点：** 可能牺牲一点响应速度（但下楼梯本来就不需要快速响应）

---

### 方案 2：降低前进速度
修改 `deploy_single_step/configs/g1_single_step_down.yaml`：

```yaml
# 原配置
command_range:
  lin_vel_x: [0.2, 0.6]

# 修改为（更保守，优先稳定性）
command_range:
  lin_vel_x: [0.15, 0.45]
```

**原理：** 更慢的速度 → 落地冲击更小 → 角动量更容易控制
**缺点：** 任务完成时间变长

---

### 方案 3：组合方案（最佳平衡）
同时应用方案1和方案2：
- Kd 降到 `[3, 3, 3, ...]`（吸收冲击）
- 速度范围 `[0.2, 0.5]`（不过分降低速度，保留任务效率）

---

## 重训练方案（根本解决）

如果快速修复效果不够，需要重新训练一个"稳定版"下楼梯策略。

### 修改 `unitree_rl_mjlab/src/tasks/velocity/config/g1_single_step_down/env_cfgs.py`

在 `unitree_g1_single_step_down_env_cfg()` 函数的 reward 配置部分：

```python
# ===== 恢复部分稳定性 reward =====
# 原配置：删除了 soft_landing 和 angular_momentum

# 1. 恢复 soft_landing（但权重降低，不要过度惩罚）
cfg.rewards["soft_landing"] = RewardTermCfg(
    func=task_mdp.soft_landing,
    weight=-0.5,  # 原默认 -1.0，降低到 -0.5：惩罚落地冲击但不过分
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        "threshold": 30.0,  # N，超过此力视为"硬着陆"
    },
)

# 2. 恢复 angular_momentum（但权重很低，只约束极端情况）
cfg.rewards["angular_momentum"] = RewardTermCfg(
    func=task_mdp.angular_momentum_l2,
    weight=-0.05,  # 非常低的权重，只防止"失控旋转"
    params={"asset_cfg": SceneEntityCfg("robot")},
)

# 3. 提高 body_orientation 权重（从 -0.3 恢复到 -0.6）
cfg.rewards["body_orientation_l2"].weight = -0.6  # 原 -0.3 → -0.6，更重视躯干姿态
```

### 为什么这样改有效

| 改动 | 效果 | 权衡 |
|------|------|------|
| 恢复 soft_landing (weight=-0.5) | 策略学会"弯腿缓冲"落地冲击 | 可能让机器人略微"怕下台阶"，但-0.5比原-1.0温和 |
| 恢复 angular_momentum (weight=-0.05) | 防止落地后躯干旋转失控 | 极低权重，不影响正常步态 |
| body_orientation: -0.3 → -0.6 | 落地后更快恢复直立 | 仍允许适度前倾（比上楼梯的-1.0宽松） |

### 训练命令
```bash
cd /home/huanghb/workspace/Unitree-G1-Single-Step-Up-Down/unitree_rl_mjlab
uv run python scripts/train.py \
  --task velocity-unitree_g1_single_step_down \
  --headless \
  --num_envs 8192 \
  --max_iterations 20000
```

### 预期效果
- 落地瞬间**弯腿缓冲**，不是僵硬直腿着地
- 躯干晃动幅度 < 5°，1步内恢复稳定（原需要2-3步）
- 仍然敢下台阶，不会出现"悬停不敢迈"的情况

---

## 调试技巧

### 如果重训练后变得"不敢下台阶"
说明 soft_landing 惩罚过重，进一步降低：
```python
cfg.rewards["soft_landing"].weight = -0.3  # -0.5 → -0.3
```

### 如果仍然晃动
1. **检查 foot_slip 权重**：
   ```python
   cfg.rewards["foot_slip"].weight = -0.2  # 原 -0.1 → -0.2，减少滑步
   ```
2. **查看训练日志的 body_orientation_l2**：
   ```bash
   tensorboard --logdir logs/rsl_rl/g1_single_step_down
   ```
   如果曲线始终是大负数（< -10），说明躯干始终倾斜，需要提高 body_orientation 权重到 -1.0。

### 如果落地后有"弹跳"感
可能是 Kp 刚度过高，尝试降低膝关节 Kp：
```yaml
kps: [200, 200, 200, 150, 150, 200, 150, 150, 200, 180, 180, ...]
#                                                    ↑ 膝关节 200→180
```

---

## 推荐实施顺序

1. **先试方案1（降 Kd）** → 改 yaml，立即测试，耗时 < 5 分钟
2. **如果仍晃动，加上方案2（降速度）** → 再改 yaml，再测试
3. **如果效果仍不满意，考虑重训练** → 修改 env_cfgs.py，训练约 6-8 小时

大多数情况下，方案1+2 的组合就能解决问题喵～ ฅ'ω'ฅ
