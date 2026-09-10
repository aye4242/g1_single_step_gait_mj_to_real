#!/usr/bin/env python3
"""
从最佳训练 checkpoint 导出 ONNX 模型
作者：浮浮酱 (猫娘工程师)
日期：2026-09-09

简化版：直接加载 actor 模型并导出，无需重建完整环境
"""

import os
import torch
import sys

print("=" * 80)
print("🎯 最佳模型 ONNX 导出工具")
print("=" * 80)

# 配置
CHECKPOINT_PATH = "logs/rsl_rl/g1_single_step_up/2026-09-09_03-14-47/model_10000.pt"
OUTPUT_PATH = "logs/rsl_rl/g1_single_step_up/2026-09-09_03-14-47/policy_best.onnx"

print(f"\n📦 Checkpoint: {CHECKPOINT_PATH}")
print(f"📤 Output: {OUTPUT_PATH}\n")

# 1. 加载 checkpoint
print("⏳ Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

print(f"✅ Checkpoint keys: {list(checkpoint.keys())}")
iteration = checkpoint.get('iter', 'unknown')
print(f"✅ Training iteration: {iteration}\n")

# 2. 提取 actor state_dict
if 'actor_state_dict' not in checkpoint:
    print("❌ Error: No actor_state_dict in checkpoint")
    sys.exit(1)

actor_state = checkpoint['actor_state_dict']
print(f"✅ Actor parameters: {len(actor_state)} layers\n")

# 显示网络结构
print("📊 Actor Network Structure:")
print("-" * 80)
total_params = 0
for key, tensor in actor_state.items():
    params = tensor.numel()
    total_params += params
    print(f"  {key:50s} {str(list(tensor.shape)):25s} = {params:,} params")
print(f"\n  Total: {total_params:,} parameters ({total_params/1e6:.3f}M)\n")

# 3. 重建 actor 网络
print("⏳ Reconstructing actor network...")

# 添加项目路径
sys.path.insert(0, 'unitree_rl_mjlab/src')
sys.path.insert(0, 'unitree_rl_mjlab')

from rsl_rl.models import MLPModel

# 从 state_dict 推断网络结构
input_dim = actor_state['mlp.0.weight'].shape[1]  # 490
output_dim = actor_state['mlp.6.weight'].shape[0]  # 29
hidden_dims = [512, 256, 128]  # 固定结构（从 config 已知）

print(f"  Input dim: {input_dim}")
print(f"  Hidden dims: {hidden_dims}")
print(f"  Output dim: {output_dim}\n")

# 构建模型
actor = MLPModel(
    obs_shape=input_dim,
    action_shape=output_dim,
    hidden_dims=hidden_dims,
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
)

# 加载权重
actor.load_state_dict(actor_state)
print("✅ Actor weights loaded\n")

# 4. 准备导出
print("⏳ Preparing ONNX export...")
actor_onnx = actor.as_onnx()
actor_onnx.to("cpu")
actor_onnx.eval()

# 5. 导出 ONNX
print("⏳ Exporting to ONNX...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

torch.onnx.export(
    actor_onnx,
    actor_onnx.get_dummy_inputs(),
    OUTPUT_PATH,
    export_params=True,
    opset_version=18,
    verbose=False,
    input_names=actor_onnx.input_names,
    output_names=actor_onnx.output_names,
    dynamic_axes={},
    dynamo=False,
)

print(f"✅ ONNX exported to: {OUTPUT_PATH}\n")

# 6. 验证 ONNX
print("⏳ Verifying ONNX model...")
import onnx
onnx_model = onnx.load(OUTPUT_PATH)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid\n")

# 7. 显示 ONNX 信息
print("=" * 80)
print("📊 ONNX Model Information")
print("=" * 80)

for inp in onnx_model.graph.input:
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    print(f"Input: {inp.name}")
    print(f"  Shape: {shape}")
    print(f"  Type: float32\n")

for out in onnx_model.graph.output:
    shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    print(f"Output: {out.name}")
    print(f"  Shape: {shape}")
    print(f"  Type: float32\n")

file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"File size: {file_size:.2f} MB")

print("\n" + "=" * 80)
print("✅ Export completed successfully!")
print("=" * 80)
print(f"\n📦 Best model (iteration {iteration}) exported to:")
print(f"   {OUTPUT_PATH}")
print("\n🎉 Ready for deployment!\n")

