#!/usr/bin/env python3
"""
Unitree G1 单台阶上楼任务真机部署脚本
========================================

基于 unitree_rl_mjlab 训练的盲走策略 (ONNX 推理)
作者：浮浮酱 (猫娘工程师)
日期：2026-09-09

特性：
- ONNX 推理 (轻量、跨平台)
- 5 帧历史观测 (490 维输入)
- 盲走策略 (无高度扫描)
- 步态相位奖励
- 安全监控

使用方法：
    python deploy_single_step.py --net eno1 --config configs/g1_single_step.yaml
"""

import sys
import time
from threading import Lock
from pathlib import Path

import numpy as np
import onnxruntime as ort
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import (
    MotorMode,
    create_damping_cmd,
    init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.safety_monitor import SafetyMonitor
from config_single_step import ConfigSingleStep


class SingleStepController:
    """单台阶上楼控制器 (ONNX 推理版本)"""

    def __init__(self, config: ConfigSingleStep, net: str) -> None:
        print("=" * 80)
        print("🚀 Unitree G1 单台阶上楼真机部署")
        print("=" * 80)
        print(f"策略文件: {config.policy_path}")
        print(f"网络接口: {net}")
        print(f"控制频率: {1/config.control_dt:.1f} Hz")
        print("=" * 80)

        ChannelFactoryInitialize(0, net)

        self.first_run = True
        self.config = config
        self.remote_controller = RemoteController()

        # 加载 ONNX 策略
        self.load_onnx_policy()

        # 初始化线程
        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)
        self.cmd_lock = Lock()

        # 机器人状态
        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        # 当前观测 (98 维)
        self.current_obs = np.zeros(config.num_obs, dtype=np.float32)

        # 5 帧历史观测 (5 × 98 = 490 维)
        self.obs_history = np.zeros(
            (config.history_length, config.num_obs),
            dtype=np.float32
        )

        # 步态相位追踪
        self.gait_phase_time = 0.0

        # 安全监控器
        self.safety_monitor = SafetyMonitor(config)

        # 速度指令限制
        self.clip_min_command = np.array([
            self.config.command_range["lin_vel_x"][0],
            self.config.command_range["lin_vel_y"][0],
            self.config.command_range["ang_vel_z"][0],
        ], dtype=np.float32)
        self.clip_max_command = np.array([
            self.config.command_range["lin_vel_x"][1],
            self.config.command_range["lin_vel_y"][1],
            self.config.command_range["ang_vel_z"][1],
        ], dtype=np.float32)

        # 预热策略
        print("\n[INFO] 预热 ONNX 推理引擎...")
        for _ in range(50):
            dummy_input = self.obs_history.flatten().reshape(1, -1).astype(np.float32)
            self.onnx_session.run(None, {self.input_name: dummy_input})
        print("[INFO] 预热完成\n")

        # 初始化通信
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        # 启动流程
        self.wait_for_low_state()
        self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        self.publish_thread.Start()
        self.wait_for_start()

        self.move_to_default_pos()
        self.wait_for_control()

        print("\n" + "=" * 80)
        print("🎮 开始控制！")
        print("=" * 80)
        print("控制说明:")
        print("  左摇杆: 控制前进/后退、左右平移")
        print("  右摇杆: 控制转向")
        print("  A 按钮: 开始控制 (已按下)")
        print("  SELECT: 紧急停止")
        print("=" * 80 + "\n")

        self.run_thread.Start()

    def load_onnx_policy(self):
        """加载 ONNX 策略"""
        policy_path = Path(self.config.policy_path)

        if not policy_path.exists():
            raise FileNotFoundError(f"策略文件不存在: {policy_path}")

        print(f"\n[INFO] 加载 ONNX 模型: {policy_path}")

        # 创建 ONNX Runtime session
        providers = ['CPUExecutionProvider']  # 可选: CUDAExecutionProvider
        self.onnx_session = ort.InferenceSession(
            str(policy_path),
            providers=providers
        )

        # 获取输入输出名称
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        # 验证输入输出形状
        input_shape = self.onnx_session.get_inputs()[0].shape
        output_shape = self.onnx_session.get_outputs()[0].shape

        expected_input = self.config.history_length * self.config.num_obs
        expected_output = self.config.num_actions

        print(f"[INFO] 输入: {self.input_name} {input_shape}")
        print(f"[INFO] 输出: {self.output_name} {output_shape}")

        if input_shape[-1] != expected_input:
            raise ValueError(f"输入维度不匹配: 期望 {expected_input}, 实际 {input_shape[-1]}")
        if output_shape[-1] != expected_output:
            raise ValueError(f"输出维度不匹配: 期望 {expected_output}, 实际 {output_shape[-1]}")

        print(f"[INFO] 推理引擎: {self.onnx_session.get_providers()}")
        print("[INFO] ONNX 模型加载成功\n")

    def LowStateHandler(self, msg):
        """接收机器人状态"""
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def publish(self):
        """发布控制命令 (500Hz)"""
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def stop(self):
        """紧急停止"""
        print("\n[警告] SELECT 按钮检测到，执行紧急停止!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)

    def wait_for_low_state(self):
        """等待连接到机器人"""
        print("[INFO] 正在连接到机器人...")
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("[INFO] ✅ 成功连接到机器人\n")

    def wait_for_start(self):
        """等待 START 按钮开始"""
        print("[INFO] 进入零力矩状态")
        print("[INFO] 请按 START 按钮移动到默认位置...")
        while self.remote_controller.button[KeyMap.start] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)
        print("[INFO] ✅ START 按钮已按下\n")

    def move_to_default_pos(self):
        """平滑移动到默认站立姿态"""
        print("[INFO] 正在移动到默认姿态...")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.joint2motor_idx
        dof_size = len(dof_idx)

        # 记录初始位置
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # 线性插值到默认位置
        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()

            alpha = i / num_step
            with self.cmd_lock:
                for j in range(dof_size):
                    motor_idx = dof_idx[j]
                    target_pos = self.config.default_joint_pos[j]
                    self.low_cmd.motor_cmd[motor_idx].q = (
                        init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    )
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
            time.sleep(self.config.control_dt)

        print("[INFO] ✅ 已到达默认姿态\n")

    def wait_for_control(self):
        """等待 A 按钮开始控制"""
        print("[INFO] 进入默认姿态状态")
        print("[INFO] 请按 A 按钮开始策略控制...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)
        # 重置相位和历史，对齐训练时 episode reset 的行为
        self.gait_phase_time = 0.0
        self.obs_history = np.zeros(
            (self.config.history_length, self.config.num_obs), dtype=np.float32
        )
        self.first_run = True
        print("[INFO] ✅ A 按钮已按下，开始策略控制\n")

    def compute_gait_phase(self) -> np.ndarray:
        """计算步态相位 (2 维: sin, cos)"""
        period = self.config.gait_phase_period  # 0.8s
        phase = (self.gait_phase_time % period) / period  # [0, 1)

        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        return np.array([sin_phase, cos_phase], dtype=np.float32)

    def run(self):
        """主控制循环 (50Hz)"""
        # 1. 读取关节状态
        for i in range(len(self.config.joint2motor_idx)):
            motor_idx = self.config.joint2motor_idx[i]
            self.joint_pos[i] = self.low_state.motor_state[motor_idx].q
            self.joint_vel[i] = self.low_state.motor_state[motor_idx].dq

        # 2. 读取 IMU
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # 如果使用腰部 IMU，需要变换到 pelvis 坐标系
        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel
            )

        gravity_orientation = get_gravity_orientation(quat)

        # 3. 处理关节观测
        joint_pos = (self.joint_pos - self.config.default_joint_pos) * self.config.dof_pos_scale
        joint_vel = self.joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        # 4. 速度指令（从遥控器）
        command = np.array(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx],
            dtype=np.float32
        )
        command *= self.config.command_scale
        command = np.clip(command, self.clip_min_command, self.clip_max_command)

        # 5. 构建当前观测 (98 维)
        # [ang_vel(3), gravity(3), command(3), phase(2), joint_pos(29), joint_vel(29), action(29)]
        num_actions = self.config.num_actions
        self.current_obs[:3] = ang_vel
        self.current_obs[3:6] = gravity_orientation
        self.current_obs[6:9] = command

        # 步态相位：静止时归零（与训练一致）
        cmd_speed = np.linalg.norm(command[:2])
        gait_phase = self.compute_gait_phase()
        if cmd_speed < 0.1:
            gait_phase = np.zeros(2, dtype=np.float32)
        self.current_obs[9:11] = gait_phase
        self.gait_phase_time += self.config.control_dt

        # 关节状态
        self.current_obs[11 : 11 + num_actions] = joint_pos
        self.current_obs[11 + num_actions : 11 + num_actions * 2] = joint_vel
        self.current_obs[11 + num_actions * 2 : 11 + num_actions * 3] = self.action

        # 6. 更新观测历史
        if self.first_run:
            # 第一次运行，用当前观测填充整个历史
            self.obs_history[:] = self.current_obs.reshape(1, -1)
            self.first_run = False
        else:
            # 滚动更新 (FIFO)
            self.obs_history = np.concatenate(
                (self.obs_history[1:], self.current_obs.reshape(1, -1)),
                axis=0
            )

        # 7. ONNX 推理
        obs_input = self.obs_history.flatten().reshape(1, -1).astype(np.float32)
        obs_input = np.clip(obs_input, -100, 100)  # 防止异常值

        action_output = self.onnx_session.run(
            [self.output_name],
            {self.input_name: obs_input}
        )[0]

        self.action = action_output.squeeze()
        self.action = np.clip(self.action, -100, 100)

        # 8. 安全检查
        if not self.safety_monitor.check_safety(self.low_state, self.action):
            print("[警告] 安全检查失败，执行阻尼控制!")
            with self.cmd_lock:
                self.low_cmd = create_damping_cmd(self.low_cmd)
            return

        # 9. 计算目标位置并发送命令
        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale

        with self.cmd_lock:
            for i in range(len(self.config.joint2motor_idx)):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unitree G1 单台阶上楼真机部署")
    parser.add_argument("--net", type=str, default="eno1", help="网络接口 (默认: eno1)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/g1_single_step.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()

    # 加载配置
    config = ConfigSingleStep(args.config)

    # 创建控制器
    controller = SingleStepController(config, args.net)

    # 主循环
    try:
        while True:
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("\n[INFO] SELECT 按钮检测到，准备退出...")
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[INFO] 检测到 Ctrl+C，准备退出...")
    finally:
        # 清理
        controller.run_thread.Wait()
        controller.publish_thread.Wait()
        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)
        time.sleep(0.2)
        print("[INFO] 退出完成")


if __name__ == "__main__":
    main()
