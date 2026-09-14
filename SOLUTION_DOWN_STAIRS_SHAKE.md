# 下楼梯落地摇晃问题 - 重训练方案

## 问题根源
训练时删除了 `soft_landing` 和 `angular_momentum` reward，策略没学会稳定落地。

## 重训练配置修改

修改 `unitree_rl_mjlab/src/tasks/velocity/config/g1_single_step_down/env_cfgs.py`：

```python
# 不要完全删除 soft_landing，改为降权
cfg.rewards["soft_landing"].weight = -5e-4  # 原 -1e-3 太重，改轻一点

# 恢复 angular_momentum，但权重降低
cfg.rewards["angular_momentum"] = RewardTermCfg(
    func=task_mdp.angular_momentum_penalty,
    weight=-0.01,  # 原 -0.025，下楼梯时适当放松
    params={"sensor_name": "robot/root_angmom"},
)

# body_orientation 权重可以稍微提高
cfg.rewards["body_orientation_l2"].weight = -0.5  # -0.3 → -0.5

# 增加躯干角速度惩罚
cfg.rewards["body_ang_vel"].weight = -0.08  # 原基础配置是 -0.05
```

## 训练命令
```bash
cd unitree_rl_mjlab
uv run python scripts/train.py \
  --task velocity-unitree_g1_single_step_down \
  --headless \
  --num_envs 8192 \
  --max_iterations 20000
```

## 预期效果
- 落地时膝盖主动柔顺缓冲（soft_landing）
- 躯干角动量受控，不会大幅摆动（angular_momentum）
- 整体姿态更稳定（body_orientation 权重提高）
