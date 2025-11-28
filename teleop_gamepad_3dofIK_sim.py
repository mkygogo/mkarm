import time
import numpy as np
import pygame
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import sys
import os
import logging
from datetime import datetime

# --- 1. 全局配置 ---
URDF_PATH = "hardware/urdf/urdf/dk2.SLDASM.urdf"
MESH_DIR = "hardware/urdf" 
FREQ = 60 
TRANS_SPEED = 0.005   # 末端移动速度 (XYZ)
JOINT_SPEED = 0.005    # 关节旋转速度 (J4-J6)
GRIPPER_SPEED = 0.002 # 夹爪速度
LONG_PRESS_TIME = 2.0 #长按判定时间（秒）

# 空间限制参数
MAX_RADIUS = 0.5      
MIN_RADIUS_XY = 0.05  
MIN_Z = 0.0            
MAX_Y = 0  

# 手柄按键映射 (Xbox Controller)
BTN_A = 0
BTN_B = 1
BTN_X = 2  # Reset
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5
AXIS_LX = 0
AXIS_LY = 1
AXIS_RX = 3 
AXIS_RY = 4 
AXIS_LT = 2
AXIS_RT = 5
HAT_ID = 0

# --- 日志设置 ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. IK 解算器 (只负责 J1-J3)
# ==========================================
class ThreeDofIKSolver:
    def __init__(self, model, data, frame_id, joint_limits):
        self.model = model
        self.data = data
        self.frame_id = frame_id
        self.joint_limits = joint_limits 
        
        self.max_iter = 15
        self.tol = 1e-3
        self.w_bias = 0.05
        self.q_ref_3dof = np.array([0.0, 1.5, -1.0]) 

    def solve(self, target_pos, q_current, dt=0.1):
        q = q_current.copy()
        debug_info = ""
        cond = 1.0
        final_err = 0.0
        success = False
        
        for i in range(self.max_iter):
            pin.framesForwardKinematics(self.model, self.data, q)
            current_pos = self.data.oMf[self.frame_id].translation
            
            err = target_pos - current_pos
            final_err = np.linalg.norm(err)
            
            if final_err < self.tol:
                success = True
                debug_info = "✅ Reached"
                break
            
            J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_sub = J[:3, :3] 
            
            cond = np.linalg.cond(J_sub)
            
            damp = 1e-3 + 0.001 * (max(0, cond - 30))**2
            damp = min(damp, 0.1)

            H = J_sub.dot(J_sub.T) + damp * np.eye(3)
            v = J_sub.T.dot(np.linalg.solve(H, err))
            
            bias_force = self.w_bias * (self.q_ref_3dof - q[:3])
            v += bias_force * 0.1
            
            v = np.clip(v, -0.5, 0.5) 
            
            q[:3] += v * dt
            
            for k in range(3):
                q[k] = max(self.joint_limits[k][0], min(q[k], self.joint_limits[k][1]))
                
        if final_err > 0.05:
            debug_info = f"⛔ Diverged (Err:{final_err*100:.1f}cm)"
            success = False
        elif debug_info == "":
            debug_info = "✅ Reached"
            success = True
            
        return q, debug_info, cond, success, final_err


# ==========================================
# 2. 6自由度机械臂封装类
# ==========================================
class SixDofArm:
    def __init__(self, urdf_path, mesh_dir):
        self.model, self.collision_model, self.visual_model = self._load_model(urdf_path, mesh_dir)
        self.data = self.model.createData()
        
        self.joint_limits = [
            [-3.0, 3.0],   # J1
            [-0.3, 3.0],   # J2
            [-3.0, 0.0],   # J3
            [-1.7, 1.2],   # J4
            [-0.7, 0.7],   # J5
            [-2.0, 2.0],   # J6
            [0.0, 0.04],   # Gripper
        ]
        
        if self.model.existFrame("link4"):
            self.ik_frame_id = self.model.getFrameId("link4")
        else:
            self.ik_frame_id = self.model.getFrameId("link3")
            
        self.ik_solver = ThreeDofIKSolver(self.model, self.data, self.ik_frame_id, self.joint_limits[:3])
        
        # --- [修改点 1] 初始化关节姿态 ---
        self.q = pin.neutral(self.model)
        # J1-J3
        self.q[0] = 0.020
        self.q[1] = 1.671  
        self.q[2] = -0.670
        self.q[3] = -1.20
        self.q[4] = 0.0
        self.q[5] = 0.0
        
        #  零位模式标志：如果是 True，则暂时忽略最小半径限制
        self.in_zero_mode = False

        # 同步目标位置
        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy() 

        self.running = True
        # 用于记录 X 键的状态
        self.x_press_start_time = None  # 按下的开始时间
        self.zero_reset_done = False    # 标记是否已经触发过全归零

    def _load_model(self, urdf_path, mesh_dir):
        abs_urdf_path = os.path.abspath(urdf_path)
        abs_mesh_dir = os.path.abspath(mesh_dir)
        meshes_folder_abs = os.path.join(abs_mesh_dir, "meshes")
        
        with open(abs_urdf_path, 'r') as f: urdf_content = f.read()
        urdf_content = urdf_content.replace('filename="package://dk2.SLDASM/meshes/', f'filename="{meshes_folder_abs}/')
        urdf_content = urdf_content.replace('filename="../meshes/', f'filename="{meshes_folder_abs}/')
        
        model = pin.buildModelFromXML(urdf_content)
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf', delete=False) as tmp:
            tmp.write(urdf_content)
            tmp_urdf_path = tmp.name
        try:
            visual_model = pin.buildGeomFromUrdf(model, tmp_urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_dir)
            collision_model = pin.buildGeomFromUrdf(model, tmp_urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)
        finally:
            os.remove(tmp_urdf_path)
            
        return model, collision_model, visual_model

    def reset(self):
        """ 强制复位到初始状态 """
        self.q = pin.neutral(self.model)
        # --- [修改点 2] 复位时的关节姿态 ---
        self.q[0] = 0.020
        self.q[1] = 1.671  
        self.q[2] = -0.670
        self.q[3] = -1.20  # J4
        self.q[4] = 0.0    # J5
        self.q[5] = 0.0    # J6
        self.q[6:] = 0.0   # Gripper
        
        # 退出零位模式
        self.in_zero_mode = False
        #恢复 IK 的偏置目标为标准姿态
        # 这样 IK 闲置时会自动把机械臂维持在这个姿态附近
        self.ik_solver.q_ref_3dof = np.array([0.0, 1.5, -1.0])

        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy()
        logger.info("🔄 Arm Reset Triggered (J4=-1.20)")

    def reset_to_zero(self):
        """ 强制归零：所有关节设为 0 """
        self.q = np.zeros(self.model.nq) # 强制全 0
        
        #进入零位模式，防止下一帧 update 被强制推开
        #将 IK 的偏置目标也设为 0
        # 否则 IK 会试图把 J2/J3 拉回 1.5/-1.0 的位置
        self.in_zero_mode = True
        self.ik_solver.q_ref_3dof = np.array([0.0, 0.0, 0.0])

        # 必须同步更新 target_pos，否则下一帧 IK 会把机械臂猛拽回去
        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy()
        logger.info("⚠️ ALL JOINTS ZEROED (0,0,0...)")

    def update(self, xyz_delta, manual_controls, dt=0.1):
        # 1. 检测是否有用户输入，如果有，则退出“零位模式”
        has_input = np.linalg.norm(xyz_delta) > 1e-6 or any(val != 0 for val in manual_controls.values())
        if has_input:
            self.in_zero_mode = False

        # ... (关节直接控制逻辑保持不变) ...
        if manual_controls['j4'] != 0:
            self.q[3] += manual_controls['j4'] * JOINT_SPEED
            self.q[3] = np.clip(self.q[3], self.joint_limits[3][0], self.joint_limits[3][1])
        if manual_controls['j5'] != 0:
            self.q[4] += manual_controls['j5'] * JOINT_SPEED
            self.q[4] = np.clip(self.q[4], self.joint_limits[4][0], self.joint_limits[4][1])
        if manual_controls['j6'] != 0:
            self.q[5] += manual_controls['j6'] * JOINT_SPEED
            self.q[5] = np.clip(self.q[5], self.joint_limits[5][0], self.joint_limits[5][1])
        if manual_controls['gripper'] != 0:
            delta = manual_controls['gripper'] * GRIPPER_SPEED
            if len(self.q) > 6:
                self.q[6] += delta 
                self.q[6] = np.clip(self.q[6], self.joint_limits[6][0], self.joint_limits[6][1])
            if len(self.q) > 7:
                self.q[7] += delta 
                self.q[7] = np.clip(self.q[7], self.joint_limits[6][0], self.joint_limits[6][1])

        # 2. 处理 XYZ IK
        self.target_pos += xyz_delta
        
        clamped_msg = ""
        old_pos = self.target_pos.copy()
        
        # [核心修改] 只有当不在零位模式时，才执行空间限制检查
        if not self.in_zero_mode:
            if self.target_pos[1] > MAX_Y: self.target_pos[1] = MAX_Y
            if self.target_pos[2] < MIN_Z: self.target_pos[2] = MIN_Z
            
            xy_dist = np.linalg.norm(self.target_pos[:2])
            if xy_dist < MIN_RADIUS_XY:
                if xy_dist < 1e-6: self.target_pos[:2] = [0, -MIN_RADIUS_XY]
                else: self.target_pos[:2] *= (MIN_RADIUS_XY / xy_dist)
                
            dist = np.linalg.norm(self.target_pos)
            if dist > MAX_RADIUS:
                self.target_pos *= (MAX_RADIUS / dist)
                
            if not np.array_equal(old_pos, self.target_pos):
                clamped_msg = "🔒 Clamped"
        else:
            clamped_msg = "⚠️ Zero Mode"

        q_new, debug_msg, cond, success, err = self.ik_solver.solve(self.target_pos, self.q)
        
        if not success:
            # 如果在零位模式下 IK 失败（通常不会），不要回退，因为 valid_pos 可能在限制区外
            if not self.in_zero_mode: 
                self.target_pos = self.valid_target_pos.copy()
                self.target_pos *= 0.99 
                self.valid_target_pos = self.target_pos.copy()
                debug_msg += " -> AUTO-RETREAT"
        else:
            self.q = q_new
            if err < 0.02:
                self.valid_target_pos = self.target_pos.copy()
                
        return debug_msg, cond, clamped_msg


# ==========================================
# 3. 6自由度仿真封装类
# ==========================================
class SixDofSim:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.js = None
        if pygame.joystick.get_count() > 0:
            self.js = pygame.joystick.Joystick(0)
            self.js.init()
            logger.info(f"🎮 Joystick: {self.js.get_name()}")
        
        self.arm = SixDofArm(URDF_PATH, MESH_DIR)
        
        self.viz = MeshcatVisualizer(self.arm.model, self.arm.collision_model, self.arm.visual_model)
        try:
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
        except: pass
        
        self._init_visuals()
        self.clock = pygame.time.Clock()
        self.running = True

    def _init_visuals(self):
        self.viz.viewer["target"].set_object(g.Sphere(0.04), g.MeshBasicMaterial(color=0xff0000, opacity=0.8))
        self.viz.viewer["workspace_outer"].set_object(g.Sphere(MAX_RADIUS), g.MeshBasicMaterial(color=0xffffff, opacity=0.1, wireframe=False))
        cyl_geom = g.Cylinder(0.4, MIN_RADIUS_XY, MIN_RADIUS_XY)
        self.viz.viewer["workspace_inner"].set_object(cyl_geom, g.MeshBasicMaterial(color=0xff0000, opacity=0.3, wireframe=False))
        self.viz.viewer["workspace_inner"].set_transform(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0.2],[0,0,0,1]]))

    def _filter_stick(self, val):
        if abs(val) < 0.15: return 0.0
        return val

    def _get_inputs(self):
        pygame.event.pump()
        
        xyz_delta = np.zeros(3)
        manual = {'j4':0, 'j5':0, 'j6':0, 'gripper':0}
        
        if self.js:
            # --- [核心修改] 长按 X 键逻辑 ---
            x_btn_state = self.js.get_button(BTN_X)
            
            if x_btn_state == 1: # 按钮被按住
                if self.x_press_start_time is None:
                    # 刚按下的第一帧：执行普通复位
                    self.x_press_start_time = time.time()
                    self.arm.reset() 
                    self.zero_reset_done = False
                else:
                    # 持续按住：检查时间
                    duration = time.time() - self.x_press_start_time
                    # 如果超过阈值，且还没触发过全归零
                    if duration > LONG_PRESS_TIME and not self.zero_reset_done:
                        self.arm.reset_to_zero()
                        self.zero_reset_done = True
                
                # 按住 X 键期间，禁止移动
                return np.zeros(3), manual 
            
            else: # 按钮松开
                self.x_press_start_time = None
                self.zero_reset_done = False

            # --- 以下是正常的摇杆控制逻辑 (保持不变) ---
            lx = self._filter_stick(self.js.get_axis(AXIS_LX))
            ly = self._filter_stick(self.js.get_axis(AXIS_LY))
            ry = self._filter_stick(self.js.get_axis(AXIS_RY))
            
            xyz_delta[0] = -lx * TRANS_SPEED
            xyz_delta[1] = ly * TRANS_SPEED
            xyz_delta[2] = -ry * TRANS_SPEED
            
            rx = self._filter_stick(self.js.get_axis(AXIS_RX))
            manual['j5'] = -rx 
            
            hat = self.js.get_hat(HAT_ID)
            manual['j6'] = -hat[0] 
            manual['j4'] = -hat[1] 
            
            lt_val = (self.js.get_axis(AXIS_LT) + 1) / 2 
            rt_val = (self.js.get_axis(AXIS_RT) + 1) / 2 
            
            if rt_val > 0.1: manual['gripper'] = 1 
            elif lt_val > 0.1: manual['gripper'] = -1
            
        return xyz_delta, manual


    def run(self):
        logger.info("🚀 Simulation Loop Started")
        log_counter = 0
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
            
            xyz_delta, manual_ctrl = self._get_inputs()
            
            debug_msg, cond, clamp_msg = self.arm.update(xyz_delta, manual_ctrl)
            
            self.viz.display(self.arm.q)
            self.viz.viewer["target"].set_transform(pin.SE3(np.eye(3), self.arm.target_pos).homogeneous)
            
            info_str = (f"{debug_msg} {clamp_msg} | "
                        f"Tgt:[{self.arm.target_pos[0]:.2f}, {self.arm.target_pos[1]:.2f}, {self.arm.target_pos[2]:.2f}] | "
                        f"Joints:[{self.arm.q[0]:.2f},{self.arm.q[1]:.2f},{self.arm.q[2]:.2f},{self.arm.q[3]:.2f}, {self.arm.q[4]:.2f}, {self.arm.q[5]:.2f}]")
            
            print(info_str, end='\r')
            
            log_counter += 1
            if log_counter % 20 == 0 or "Diverged" in debug_msg:
                logger.info(info_str)
                
            self.clock.tick(FREQ)
            
        pygame.quit()
        logger.info("🛑 Simulation Ended")

if __name__ == "__main__":
    sim = SixDofSim()
    sim.run()