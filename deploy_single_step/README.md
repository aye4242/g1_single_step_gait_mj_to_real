# Unitree G1 单台阶上楼真机部署

基于 `unitree_rl_mjlab` 训练的盲走策略，用于 Unitree G1 人形机器人单台阶上楼任务。

作者：浮浮酱 (猫娘工程师)  
日期：2026-09-09

---

## 📦 目录结构

```
deploy_single_step/
├── deploy_single_step.py    # 主部署脚本
├── config_single_step.py    # 配置加载器
├── configs/
│   └── g1_single_step.yaml  # 任务配置文件
├── common/
│   ├── command_helper.py    # 命令辅助函数
│   ├── remote_controller.py # 遥控器接口
│   ├── rotation_helper.py   # 旋转变换
│   └── safety_monitor.py    # 安全监控
├── policy/
│   └── policy_best.onnx     # ONNX 策略文件（需放置）
└── README.md                # 本文档
```

---

## 🔧 依赖安装

### 1. 系统要求

- **硬件**: Unitree G1 机器人 + 控制电脑（Ubuntu 22.04）
- **网络**: 以太网连接到机器人（默认 `eno1`）
- **Python**: 3.8+

### 2. 安装 Unitree SDK2

```bash
# 克隆 SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python

# 安装依赖
pip install cyclonedds pyyaml numpy scipy

# 安装 SDK
pip install -e .
```

### 3. 安装 ONNX Runtime

```bash
# CPU 版本
pip install onnxruntime

# 或 GPU 版本（如果控制电脑有 NVIDIA GPU）
pip install onnxruntime-gpu
```

---

## 🚀 部署流程

### 步骤 1：准备策略文件

将训练好的 ONNX 模型放置到 `policy/` 目录：

```bash
mkdir -p policy
cp /path/to/policy_best.onnx policy/
```

**策略文件来源：**
- 从训练日志导出：`unitree_rl_mjlab/logs/rsl_rl/g1_single_step_up/.../policy_best.onnx`
- 要求：输入 `[1, 490]`，输出 `[1, 29]`

### 步骤 2：检查配置文件

编辑 `configs/g1_single_step.yaml`，确认：

- `policy_path`: ONNX 文件路径
- `control_dt`: 控制频率（默认 0.02 = 50Hz）
- `command_range`: 速度指令范围
- `kps`, `kds`: PD 增益（根据机器人调优）
- `default_joint_pos`: 默认站立姿态

### 步骤 3：连接机器人

```bash
# 1. 用以太网线连接机器人和控制电脑
# 2. 配置静态 IP（示例）
sudo ifconfig eno1 192.168.123.100 netmask 255.255.255.0

# 3. 测试连接
ping 192.168.123.10  # G1 默认 IP
```

### 步骤 4：运行部署脚本

```bash
cd deploy_single_step
python deploy_single_step.py --net eno1 --config configs/g1_single_step.yaml
```

**参数说明：**
- `--net`: 网络接口名称（`eno1`, `eth0`, 等）
- `--config`: 配置文件路径

---

## 🎮 操作流程

### 启动流程

1. **运行脚本** → 看到 "正在连接到机器人..."
2. **连接成功** → 显示 "✅ 成功连接到机器人"
3. **零力矩状态** → 机器人关节松弛，可手动移动
4. **按 START** → 机器人平滑移动到默认站立姿态（2 秒）
5. **按 A 按钮** → 开始策略控制！

### 遥控器控制

| 按键/摇杆 | 功能 |
|---|---|
| **左摇杆 ↑↓** | 前进/后退 (vx) |
| **左摇杆 ←→** | 左右平移 (vy) |
| **右摇杆 ←→** | 转向 (omega_z) |
| **A 按钮** | 开始控制 |
| **SELECT** | **紧急停止** |

### 紧急停止

**按下 SELECT 按钮**，机器人立即切换到阻尼模式（零力矩），程序退出。

---

## ⚠️ 安全注意事项

### 测试前准备

1. **悬空测试**
   - 用支架悬空机器人
   - 启动脚本，观察关节运动是否合理
   - 检查是否有异常抖动

2. **支撑测试**
   - 机器人站在地面，外力辅助支撑
   - 测试站立平衡

3. **平地测试**
   - 在平整地面测试前进/转向
   - 确认步态稳定

4. **台阶测试**
   - **先测试低台阶**（5cm, 10cm）
   - 逐步提升高度（15cm, 20cm）
   - 每个高度重复 20+ 次，成功率 > 90% 再提升

### 环境要求

- ✅ **平坦的助跑区**（至少 2 米）
- ✅ **稳固的台阶**（木板/水泥，非金属）
- ✅ **软垫保护**（台阶周围放置软垫）
- ✅ **空旷空间**（避免障碍物）
- ❌ 避免湿滑地面
- ❌ 避免强光干扰 IMU

### 监控指标

部署脚本包含安全监控，会在以下情况触发保护：

- 关节速度 > 20 rad/s
- 关节力矩 > 100 N·m
- 身体倾斜 > 30°
- 关节超出软限位

**触发保护时**：机器人自动切换到阻尼模式。

---

## 🐛 故障排查

### 问题 1：无法连接到机器人

**症状**: "正在连接到机器人..." 一直等待

**解决**:
```bash
# 检查网络接口
ifconfig
# 检查 DDS 通信
unitree_dds_test
# 检查机器人电源和网线
```

### 问题 2：机器人抖动

**原因**: PD 增益不匹配

**解决**: 编辑 `configs/g1_single_step.yaml`，降低 `kps`（如从 200 降到 150）

### 问题 3：机器人摔倒

**原因**:
1. 台阶高度超过策略能力
2. 速度指令过快
3. 地面摩擦力不足

**解决**:
1. 降低台阶高度
2. 减小速度范围：`command_range.lin_vel_x: [0.2, 0.6]`
3. 更换防滑垫

### 问题 4：动作不自然

**原因**: 观测归一化参数缺失

**解决**: 确保 ONNX 模型包含 `obs_normalizer` 权重，或在配置中添加归一化参数。

---

## 📊 性能优化

### 1. 提高推理速度

```bash
# 使用 ONNX Runtime GPU
pip install onnxruntime-gpu

# 修改 deploy_single_step.py 第 175 行:
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 2. 调整控制频率

- 训练频率：50Hz (`control_dt: 0.02`)
- 可尝试：100Hz (`control_dt: 0.01`) 提高响应速度
- ⚠️ 频率过高会增加计算负担

### 3. PD 增益调优

根据实际表现微调：

- **抖动** → 降低 `kps`
- **响应慢** → 提高 `kps`
- **阻尼不足** → 提高 `kds`

---

## 📝 数据记录

### 记录训练日志

修改脚本添加日志记录：

```python
# 在 run() 函数中添加
self.log_data = {
    'time': [],
    'joint_pos': [],
    'joint_vel': [],
    'action': [],
    'command': [],
}

# 每步记录
self.log_data['time'].append(time.time())
self.log_data['joint_pos'].append(self.joint_pos.copy())
# ...

# 退出时保存
np.savez('deployment_log.npz', **self.log_data)
```

---

## 🎓 进阶：Sim-to-Real Fine-tuning

如果真机表现与仿真差距较大：

### 1. 收集真机数据

```bash
# 运行部署脚本并记录数据
python deploy_single_step.py --record True
```

### 2. 在真机数据上微调

```python
# 固定大部分层，仅微调输出层
for param in model.parameters():
    param.requires_grad = False
model.fc_out.weight.requires_grad = True

# 低学习率训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

### 3. 域随机化增强

在仿真中增加：
- 更大的摩擦系数范围
- 更大的质量/惯量随机化
- 更大的观测噪声

---

## 📚 参考资料

- [Unitree SDK2 文档](https://support.unitree.com/)
- [G1DWAQ_Lab 项目](https://github.com/TienKung-Lab/G1DWAQ_Lab)
- [RSL-RL 文档](https://github.com/leggedrobotics/rsl_rl)
- [ONNX Runtime 文档](https://onnxruntime.ai/)

---

## 🙋 问题反馈

如有问题，请检查：

1. SDK 版本是否最新
2. Python 依赖是否完整
3. 配置文件参数是否正确
4. ONNX 模型是否匹配任务

---

**祝部署成功喵～** ฅ'ω'ฅ

作者：浮浮酱  
邮箱：（主人填写）  
项目：Unitree G1 Single Step Up/Down
