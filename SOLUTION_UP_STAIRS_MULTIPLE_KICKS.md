# 上楼梯多次踢台阶才能上去 - 解决方案

## 问题现象
机器人踢到台阶后需要多次频繁踢腿才能成功上台阶，不能一次性流畅上楼。

## 根本原因
1. **感知-执行延迟链**：盲策略通过 joint_vel spike 感知碰撞 → 5帧历史窗口100ms + 系统延迟4ms → 从踢到到抬腿约 **100-150ms 延迟**
2. **foot_clearance 权重不足**：训练时 weight=-1.0，相比其他主要reward（track_vel=1.0, alive=0.15, pose=1.0）占比太小
3. **台阶高度 jitter=0.1**：10%的随机化让策略学到"保守抬腿"（宁可多踢几次也不抬太高浪费能量）

## 已实施方案

### 方案 1：提高前进速度（已修改 g1_single_step.yaml）
```yaml
command_range:
  lin_vel_x: [0.35, 0.85]  # 原 0.2~1.0，避免过慢导致多次试探
```
**原理：** 更高的动量 → 踢到台阶时 joint_vel spike 更明显 → 策略更快感知到障碍。

---

## 重训练方案（根本解决）

如果速度调整效果不够，需要修改训练配置重新训练：

### 修改 `unitree_rl_mjlab/src/tasks/velocity/config/g1_single_step/env_cfgs.py`

在 `unitree_g1_single_step_up_env_cfg()` 函数末尾 `return cfg` 之前添加：

```python
# ===== 增强上楼梯的 foot_clearance =====
# 原配置：weight=-1.0, target_height=0.10m
# 问题：权重太低，策略不重视抬腿高度
cfg.rewards["foot_clearance"].weight = -2.0  # -1.0 → -2.0，加倍惩罚
cfg.rewards["foot_clearance"].params["target_height"] = 0.12  # 0.10 → 0.12m，抬更高

# 同时提高 foot_gait 权重，确保交替步态清晰
if "foot_gait" in cfg.rewards:
    cfg.rewards["foot_gait"].weight = 1.0  # 原 0.5 → 1.0，强化步态节奏
```

### 为什么这样改有效

| 改动 | 效果 |
|------|------|
| `foot_clearance` weight: -1.0 → -2.0 | 策略更重视抬腿高度，不敢"偷懒"低抬腿 |
| `target_height`: 0.10 → 0.12m | 抬腿目标提高20%，给10%的jitter留更多余量 |
| `foot_gait` weight: 0.5 → 1.0 | 步态节奏更清晰，减少"连续踢"的混乱模式 |

### 训练命令
```bash
cd /home/huanghb/workspace/Unitree-G1-Single-Step-Up-Down/unitree_rl_mjlab
uv run python scripts/train.py \
  --task velocity-unitree_g1_single_step \
  --headless \
  --num_envs 8192 \
  --max_iterations 20000
```

### 预期效果
- 第一次踢到台阶后，**下一步直接高抬腿跨上去**，不再需要多次试探
- 步态节奏更清晰，抬腿幅度一致性更好

---

## 调试技巧

### 如果重训练后仍然多次踢腿
1. **检查 play 模式的台阶高度**：
   ```python
   # env_cfgs.py 里 play 模式
   gen.difficulty_range = (0.8, 1.0)  # 如果太高（>0.23m），改为 (0.5, 0.8)
   ```
2. **查看训练日志的 foot_clearance reward**：
   ```bash
   tensorboard --logdir logs/rsl_rl/g1_single_step
   ```
   如果 foot_clearance 曲线始终是大负数（< -5），说明策略没学会，需要进一步降低惩罚权重到 -1.5。

### 如果一次就能上但动作不流畅
可能是 foot_gait 权重过高，适当降回 0.7 左右。
