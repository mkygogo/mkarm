import pinocchio as pin
import numpy as np
import os

class MKArmIK:
    def __init__(self, urdf_path, end_effector_frame="gripper"):
        # 加载模型
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 获取末端执行器的 Frame ID
        if end_effector_frame not in self.model.names:
             # 如果找不到 gripper，尝试找最后一个关节的 link
            print(f"Warning: Frame '{end_effector_frame}' not found. Using last frame: {self.model.names[-1]}")
            self.frame_id = self.model.nframes - 1
        else:
            self.frame_id = self.model.getFrameId(end_effector_frame)

        # 初始关节配置 (q0)
        self.q_neutral = pin.neutral(self.model)
        
        # 设置关节限制 (从 URDF 读取)
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

    def solve_ik(self, target_pos, target_rot, q_guess=None, max_iter=100, eps=1e-4, dt=0.1, damp=1e-6):
        """
        使用阻尼最小二乘法 (Damped Least Squares) 求解 IK
        target_pos: [x, y, z]
        target_rot: 3x3 旋转矩阵 (np.array)
        """
        if q_guess is None:
            q_guess = self.q_neutral

        q = q_guess.copy()
        
        # 目标变换矩阵 SE3
        oMdes = pin.SE3(target_rot, np.array(target_pos))

        for i in range(max_iter):
            # 计算前向运动学
            pin.framesForwardKinematics(self.model, self.data, q)
            
            # 当前末端位姿
            dMi = oMdes.actInv(self.data.oMf[self.frame_id])
            
            # 计算误差向量 (位置 + 旋转)
            err = pin.log(dMi).vector
            
            if np.linalg.norm(err) < eps:
                # print(f"Converged in {i} iterations")
                return q, True # 成功收敛

            # 计算雅可比矩阵
            J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id)
            
            # 阻尼最小二乘: dq = J.T * (J * J.T + damp * I)^-1 * err
            # 或者更简单的形式: v = - J.pinv * err
            # 为了稳定性，这里使用简单的梯度下降步长或伪逆
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            
            # 更新关节角度
            q = pin.integrate(self.model, q, v * dt)
            
            # 强制关节限制
            q = np.clip(q, self.q_min, self.q_max)

        return q, False # 未完全收敛，但返回最佳猜测