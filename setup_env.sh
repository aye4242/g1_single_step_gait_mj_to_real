#!/usr/bin/env bash
# 在容器内执行：创建训练专用 venv 并安装 unitree_rl_mjlab 依赖
# 用法（容器内）: bash setup_env.sh
set -e

PROJ=/workspace/Unitree-G1-Single-Step-Up-Down
VENV=$PROJ/.venv-mjlab

echo "=== 1. 创建 venv（Python 3.12） ==="
uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"

echo "=== 2. 安装 PyTorch cu128（RTX 5080/Blackwell 需要） ==="
uv pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== 3. 安装 unitree_rl_mjlab（editable） ==="
cd "$PROJ/unitree_rl_mjlab"
uv pip install -e .
# 与 mjlab 1.2.0 官方 uv.lock 对齐的版本组合：
#   warp-lang 1.12.0（>=1.16 移除了 wp.context，mjlab 依赖它）
#   mujoco 3.5.0（与 mujoco-warp 3.5.0 配对；3.12 移除了 mjENBL_MULTICCD）
uv pip install "warp-lang==1.12.0" "mujoco==3.5.0" scipy

echo "=== 4. 版本记录（阶段 0 验收） ==="
python - <<'EOF'
import importlib.metadata as md
for p in ["mjlab", "mujoco-warp", "mujoco", "rsl-rl", "torch", "numpy"]:
    try:
        print(f"{p:14s} {md.version(p)}")
    except Exception:
        print(f"{p:14s} NOT FOUND")
EOF

echo "=== 5. GPU 可用性 ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== 完成。激活方式: source /workspace/Unitree-G1-Single-Step-Up-Down/.venv-mjlab/bin/activate ==="
