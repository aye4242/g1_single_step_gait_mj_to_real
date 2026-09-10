"""
单台阶任务配置加载器
作者：浮浮酱
"""

import numpy as np
import yaml
from pathlib import Path


class ConfigSingleStep:
    """单台阶上楼任务配置"""

    def __init__(self, file_path: str) -> None:
        config_path = Path(file_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # 控制参数
        self.control_dt = config["control_dt"]

        # 消息类型
        self.msg_type = config["msg_type"]  # "hg" for G1
        self.imu_type = config["imu_type"]  # "pelvis" or "torso"

        # DDS 话题
        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]

        # 策略路径
        self.policy_path = config["policy_path"]

        # 关节映射
        self.joint2motor_idx = config["joint2motor_idx"]
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)

        # 腰部关节索引（如果使用 torso IMU）
        if "torso_idx" in config:
            self.torso_idx = config["torso_idx"]

        # 缩放参数
        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = config["action_scale"]
        self.command_scale = np.array(config["command_scale"], dtype=np.float32)

        # 维度
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.history_length = config["history_length"]

        # 速度指令范围
        self.command_range = config["command_range"]

        # 步态相位配置
        self.gait_phase_period = config.get("gait_phase_period", 0.8)

        # 安全限制
        self.max_joint_vel = config.get("max_joint_vel", 20.0)
        self.max_torque = config.get("max_torque", 100.0)
        self.max_body_tilt = config.get("max_body_tilt", 30.0)  # degrees

    def __repr__(self):
        return f"ConfigSingleStep(policy={self.policy_path}, dt={self.control_dt})"
