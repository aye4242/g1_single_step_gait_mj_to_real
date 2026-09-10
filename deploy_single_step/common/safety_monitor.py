"""
安全监控器
作者：浮浮酱

监控机器人状态，防止危险动作
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class SafetyMonitor:
    """机器人安全监控器"""

    def __init__(self, config):
        self.config = config
        self.violation_count = 0
        self.max_violations = 5  # 连续 5 次违规才触发停止

    def check_safety(self, low_state, action: np.ndarray) -> bool:
        """
        检查当前状态和动作是否安全

        Args:
            low_state: 机器人状态
            action: 待执行的动作

        Returns:
            bool: True 表示安全，False 表示不安全
        """
        reasons = []

        # 1. 检查关节位置限制
        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            q = low_state.motor_state[motor_idx].q
            # G1 关节限制（简化版，实际应从配置读取）
            if q < -3.14 or q > 3.14:
                reasons.append(f"关节 {i} 超限: {q:.2f} rad")

        # 2. 检查关节速度限制
        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            dq = low_state.motor_state[motor_idx].dq
            if abs(dq) > self.config.max_joint_vel:
                reasons.append(f"关节 {i} 速度过高: {dq:.2f} rad/s")

        # 3. 检查关节力矩限制
        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            tau = low_state.motor_state[motor_idx].tau_est
            if abs(tau) > self.config.max_torque:
                reasons.append(f"关节 {i} 力矩过高: {tau:.2f} N·m")

        # 4. 检查身体姿态限制（防止摔倒）
        quat = low_state.imu_state.quaternion
        # 转换为欧拉角
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=True)
        roll, pitch, yaw = euler

        max_tilt = self.config.max_body_tilt
        if abs(roll) > max_tilt:
            reasons.append(f"Roll 角过大: {roll:.1f}°")
        if abs(pitch) > max_tilt:
            reasons.append(f"Pitch 角过大: {pitch:.1f}°")

        # 5. 检查动作变化率（防止突变）
        if np.any(np.abs(action) > 50):
            reasons.append(f"动作幅度过大: max={np.max(np.abs(action)):.2f}")

        # 判断结果
        if reasons:
            self.violation_count += 1
            if self.violation_count >= self.max_violations:
                print(f"[安全警告] 连续 {self.max_violations} 次违规:")
                for reason in reasons:
                    print(f"  - {reason}")
                return False
        else:
            # 重置计数器
            self.violation_count = max(0, self.violation_count - 1)

        return True

    def reset(self):
        """重置违规计数"""
        self.violation_count = 0
