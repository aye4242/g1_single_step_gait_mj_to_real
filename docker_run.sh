#!/usr/bin/env bash
# 启动 G1 楼梯任务训练容器（基于 mujoco-huanghb:latest）
# 用法: ./docker_run.sh          # 进入交互式 bash
#       ./docker_run.sh <cmd>    # 执行单条命令
set -e

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-mujoco-huanghb:latest}"
NAME="${NAME:-g1_rl_mjlab}"

# 同名容器已在运行：直接进入（避免误杀正在训练的容器）
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[docker_run] 容器 $NAME 已在运行，直接进入（exit 退出 shell 不会停止容器）..."
  docker exec -it "$NAME" bash
  exit 0
fi

# 清理已停止的同名残留容器
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -it --rm \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  --network host \
  --shm-size 16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/home \
  -e USER=huanghb \
  -e LOGNAME=huanghb \
  -e UV_CACHE_DIR=/tmp/uvcache \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e XAUTHORITY=/tmp/.Xauthority-host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e WANDB_MODE=offline \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/.Xauthority:/tmp/.Xauthority-host:ro" \
  -v "$PROJ_DIR:/workspace/Unitree-G1-Single-Step-Up-Down" \
  -w /workspace/Unitree-G1-Single-Step-Up-Down \
  "$IMAGE" "$@"
