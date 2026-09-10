# Unitree G1 单台阶任务 - 常用命令手册

> 环境：容器 `g1_rl_mjlab`（镜像 mujoco-huanghb:latest），venv 位于项目目录 `.venv-mjlab/`
> 宿主机项目路径：`/home/huanghb/workspace/Unitree-G1-Single-Step-Up-Down/`
> 容器内项目路径：`/workspace/Unitree-G1-Single-Step-Up-Down/`

---

## 1. 进入容器 & 激活环境

```bash
# 进入容器
docker exec -it g1_rl_mjlab bash

# 激活 venv（容器内）
cd /workspace/Unitree-G1-Single-Step-Up-Down/unitree_rl_mjlab
source ../.venv-mjlab/bin/activate
```

容器信息：IP `172.17.0.2`（docker bridge，宿主机可直接访问），无端口映射。

---

## 2. 训练

```bash
# 单台阶任务（当前正在跑）：6144 环境，预估 ~11GB 显存
python scripts/train.py Unitree-G1-Single-Step-Up --env.scene.num-envs 6144

# 上楼梯任务（已完成 20000 iter）
python scripts/train.py Unitree-G1-Stairs-Up-Blind --env.scene.num-envs 4096

# 下楼梯任务（尚未训练）
python scripts/train.py Unitree-G1-Stairs-Down-Blind --env.scene.num-envs 4096

# 查看某任务全部可覆盖参数
python scripts/train.py Unitree-G1-Single-Step-Up --help
```

> ⚠️ 注意：
> - 参数是 `--env.scene.num-envs`，**不是** `--num-envs`（train.py 用 tyro 全路径解析）
> - **一次只能跑一个训练**，两个 4096+ envs 任务同时挤一张 5080 会互相顶掉
> - 显存参考：4096 envs ≈ 7.5GB；6144 ≈ 11GB；8192 有 OOM 风险（16GB 卡）

---

## 3. 训练监控

```bash
# TensorBoard（容器内另开终端启动）
tensorboard --logdir logs/rsl_rl --port 6006 --bind_all
# 浏览器访问（宿主机）： http://172.17.0.2:6006
# 若 bridge 网络不通，宿主机做端口转发：
#   socat TCP-LISTEN:6006,fork TCP:172.17.0.2:6006  然后访问 http://localhost:6006

# 盯显存（宿主机或容器内）
watch -n 5 nvidia-smi

# 查看训练日志目录
ls -t logs/rsl_rl/<task_name>/
```

TensorBoard 重点曲线（右上角 Smoothing 拉到 0.9）：
- `Train/mean_episode_length`（第一指标，→1000 = 活满 20s）
- `Train/mean_reward`（持续上升）
- `Episode_Termination/time_out ↑ / fell_over ↓`
- `Curriculum/terrain_levels`（课程难度等级）
- `Policy/mean_std`（防躺平：episode length 低时 std 坍缩 <0.1 = 废了）

---

## 4. Play 可视化

> ⚠️ TurboVNC（display :19）下 native viewer 是 CPU 软渲染，仅 1 FPS，**不要用**。
> 必须设置 `export MUJOCO_GL=egl` 走 GPU 离屏渲染，用录像或 viser。

```bash
export MUJOCO_GL=egl

# 方式一（推荐）：录像，离线看 mp4
python scripts/play.py Unitree-G1-Single-Step-Up --num-envs 4 \
  --checkpoint-file logs/rsl_rl/g1_single_step_up/<时间戳>/model_XXX.pt \
  --video True --video-length 1000
# 视频输出在 checkpoint 同级 videos/play/ 目录，宿主机可直接打开

# 方式二：viser 网页实时交互（有摇杆可手动指挥机器人）
python scripts/play.py Unitree-G1-Single-Step-Up --num-envs 4 \
  --checkpoint-file logs/rsl_rl/g1_single_step_up/<时间戳>/model_XXX.pt \
  --viewer viser
# 浏览器访问： http://172.17.0.2:8080
```

> play 参数注意：`--video` 需显式传值 `True`；checkpoint 用 `--checkpoint-file`。
> play 模式固定 difficulty_range=(0.8,1.0)，即直接考最高难度（台阶 0.2m+）。

---

## 5. 进程/显存排障

```bash
# 训练卡死 Ctrl+C 无效时，宿主机强杀
ps aux | grep play.py     # 找 PID
kill -9 <PID>

# 查显存残留
nvidia-smi
```

---

## 6. 已完成任务的结果位置

| 任务 | checkpoint 目录 | 状态 |
|---|---|---|
| 上楼梯 stairs_up | `logs/rsl_rl/g1_stairs_up_blind/2026-09-08_08-30-47/`（model_20000.pt + policy.onnx） | ✅ 完成，episode 995/1000，地形等级 5.2/9 |
| 单台阶 single_step_up | `logs/rsl_rl/g1_single_step_up/2026-09-09_03-14-47/` | 🔄 训练中（6144 envs） |
| 下楼梯 stairs_down | `logs/rsl_rl/g1_stairs_down_blind/`（仅 model_9.pt，被顶掉的残留） | ❌ 未训 |
