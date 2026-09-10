#!/usr/bin/env python3
"""
手动导出 ONNX 策略文件
从 model_10000.pt 提取 actor 权重并导出为 ONNX 格式

作者：浮浮酱 (猫娘工程师) φ(≧ω≦*)♪
日期：2026-09-09
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ActorNetwork(nn.Module):
    """
    Actor 网络结构（与训练时一致）
    输入：490 维观测（5 帧历史）
    输出：29 维关节动作
    """
    def __init__(self):
        super().__init__()

        # 网络结构: 490 → 512 → 256 → 128 → 29
        self.fc1 = nn.Linear(490, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 29)

        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # 输出层无激活
        return x


def load_actor_from_checkpoint(ckpt_path):
    """
    从 PyTorch checkpoint 加载 actor 权重

    参数:
        ckpt_path: model_*.pt 文件路径

    返回:
        actor: 加载了权重的 ActorNetwork 模型
    """
    print(f"[INFO] 正在加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # 检查 checkpoint 结构
    if 'actor_state_dict' in checkpoint:
        actor_state = checkpoint['actor_state_dict']
        print(f"[✓] 找到 actor_state_dict，包含 {len(actor_state)} 个参数")
    else:
        raise KeyError("checkpoint 中没有 'actor_state_dict' 字段喵 >_<")

    # 创建 actor 网络
    actor = ActorNetwork()

    # 加载权重
    try:
        actor.load_state_dict(actor_state, strict=True)
        print("[✓] 成功加载所有权重！")
    except Exception as e:
        print(f"[ERROR] 权重加载失败: {e}")
        raise

    actor.eval()  # 设置为评估模式
    return actor


def export_to_onnx(actor, output_path, opset_version=11):
    """
    导出 PyTorch 模型到 ONNX 格式

    参数:
        actor: PyTorch 模型
        output_path: 输出 ONNX 文件路径
        opset_version: ONNX opset 版本（默认 11）
    """
    print(f"\n[INFO] 开始导出 ONNX，opset={opset_version}")

    # 创建虚拟输入（batch_size=1, obs_dim=490）
    dummy_input = torch.randn(1, 490)

    # 导出
    torch.onnx.export(
        actor,                          # 模型
        dummy_input,                    # 示例输入
        output_path,                    # 输出路径
        export_params=True,             # 导出参数
        opset_version=opset_version,    # ONNX 版本
        do_constant_folding=True,       # 常量折叠优化
        input_names=['observation'],    # 输入名称
        output_names=['action'],        # 输出名称
        dynamic_axes={                  # 动态维度（batch）
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )

    print(f"[✓] ONNX 导出成功: {output_path}")

    # 打印文件大小
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"[INFO] 文件大小: {file_size:.2f} MB")


def verify_onnx(onnx_path):
    """
    验证导出的 ONNX 模型

    参数:
        onnx_path: ONNX 文件路径
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[WARN] 需要安装 onnx 和 onnxruntime 来验证模型")
        print("      pip install onnx onnxruntime")
        return

    print(f"\n[INFO] 验证 ONNX 模型: {onnx_path}")

    # 1. 检查模型结构
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("[✓] ONNX 模型结构正确")

    # 2. 测试推理
    session = ort.InferenceSession(onnx_path)

    # 输入/输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"[INFO] 输入: {input_info.name}, shape={input_info.shape}, dtype={input_info.type}")
    print(f"[INFO] 输出: {output_info.name}, shape={output_info.shape}, dtype={output_info.type}")

    # 测试推理
    test_input = np.random.randn(1, 490).astype(np.float32)
    result = session.run(None, {input_info.name: test_input})
    print(f"[✓] 推理成功！输出 shape={result[0].shape}")
    print(f"[INFO] 输出范围: [{result[0].min():.3f}, {result[0].max():.3f}]")


def main():
    """主函数：导出 model_10000.pt 到 ONNX"""

    print("=" * 60)
    print("  Unitree G1 单台阶策略导出工具")
    print("  作者：浮浮酱 (猫娘工程师) ฅ'ω'ฅ")
    print("=" * 60)

    # 路径配置
    checkpoint_path = "/home/huanghb/workspace/Unitree-G1-Single-Step-Up-Down/unitree_rl_mjlab/logs/rsl_rl/g1_single_step_up/2026-09-09_03-14-47/model_10000.pt"
    output_path = "policy/policy_best.onnx"

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 步骤 1: 加载 checkpoint
    actor = load_actor_from_checkpoint(checkpoint_path)

    # 步骤 2: 导出 ONNX
    export_to_onnx(actor, output_path, opset_version=11)

    # 步骤 3: 验证模型
    verify_onnx(output_path)

    print("\n" + "=" * 60)
    print("  ✅ 导出完成喵～")
    print(f"  ONNX 文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
