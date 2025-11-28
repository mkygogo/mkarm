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
import argparse

# --- 导入真实机械臂库 ---
try:
    from follower_mkarm import MKFollower, MKFollowerConfig
    HAS_REAL_ARM_LIB = True
except ImportError:
    print("⚠️ Warning: 'follower_mkarm.py' not found. Running in Simulation-Only mode.")
    HAS_REAL_ARM_LIB = False

# --- 1. 全局配置 ---
URDF_PATH = "hardware/urdf/urdf/dk2.SLDASM.urdf"
MESH_DIR = "hardware/urdf" 
FREQ = 60 
TRANS_SPEED = 0.002   # 末端移动速度 (XYZ)
JOINT_SPEED = 0.02    # 关节旋转速度 (J4-J6)
GRIPPER_SPEED = 0.002 # 夹爪速度
LONG_PRESS_TIME = 2.0 # 长按判定时间（秒）
REAL_ARM_PORT = "/dev/ttyACM0" 

# 空间限制参数
MAX_RADIUS = 0.5      
MIN_RADIUS_XY = 0.05 #0.05  
MIN_JOINT4_Z = 0.227    # 这是Joint4/Wrist的高度，不是指尖高度      
MAX_Y = -0.05 

# 测试模式速度
JOINT_CTRL_SPEED_J1_J3 = 0.015 
JOINT_CTRL_SPEED_J4_J6 = 0.015

# -------------------------------------------------------------------------
# 硬件方向修正 (Hardware Direction Correction)
# J2=1.0, 其他=-1.0
# -------------------------------------------------------------------------
HARDWARE_DIR = {
    "joint_1": -1.0, 
    "joint_2":  1.0, 
    "joint_3": -1.0, 
    "joint_4": -1.0, 
    "joint_5": -1.0, 
    "joint_6": -1.0,
    "gripper":  1.0 
}

# -------------------------------------------------------------------------
# 手柄控制方向 (Joystick Control Direction)
# -------------------------------------------------------------------------
CONTROL_DIR = {
    # 关节直控模式
    'CTRL_J1': -1.0, 'CTRL_J2': -1.0, 'CTRL_J3': -1.0, 
    'CTRL_J4':  1.0, 'CTRL_J5':  1.0, 'CTRL_J6':  1.0,

    # IK 模式 (Sim移动方向)
    'IK_X':  -1.0, 'IK_Y': 1.0, 'IK_Z':  -1.0,
    'IK_J4': 1.0, 'IK_J5': 1.0, 'IK_J6': 1.0
}

# 真实机械臂的物理限位 (用于发送指令前的安全截断)
REAL_JOINT_LIMITS = {
    "joint_1": [-3.0, 3.0],
    "joint_2": [-0.3, 3.0],
    "joint_3": [0.0, 3.0],   # 注意：这是正值区间
    "joint_4": [-1.7, 1.2],
    "joint_5": [-0.4, 0.4],  # 范围较窄
    "joint_6": [-2.0, 2.0]
}

# 手柄按键映射 (Xbox Controller)
BTN_A = 0
BTN_B = 1
BTN_X = 2  # Reset
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5 # [Deadman Switch] 按住以激活真实机械臂控制
BTN_BACK = 6 # jointc ctrl
BTN_START = 7
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

log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_path = os.path.join(LOG_DIR, log_filename)

# 1. 创建 Logger
logger = logging.getLogger("MKArmLogger")
logger.setLevel(logging.INFO)
logger.propagate = False # 防止重复打印

# 2. 创建 Formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')

# 3. 创建 FileHandler (关键：保存引用)
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 4. 创建 StreamHandler (输出到终端)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def force_flush_log():
    """强制将缓冲区写入硬盘"""
    file_handler.flush()
    os.fsync(file_handler.stream.fileno()) # 这是一个更底层的强制写入，确保操作系统不缓存

logger.info(f"📝 Log file created at: {log_path}")
force_flush_log() # 立即刷新一次，确保文件里至少有这句话


# ==========================================
# 0. 真实机械臂接口类 (负责 Sim <-> Real 映射)
# ==========================================
class SixDofRealArm:
    def __init__(self, port):
        if not HAS_REAL_ARM_LIB:
            raise ImportError("follower_mkarm library missing")
            
        self.config = MKFollowerConfig(
            port=port,
            joint_velocity_scaling=1.0, 
            disable_torque_on_disconnect=True
        )
        self.robot = MKFollower(self.config)
        logger.info(f"🔗 Connecting to Real Arm on {port}...")
        self.robot.connect()
        logger.info("✅ Real Arm Connected!")

    def read_joints(self):
        """ 
        读取真实机械臂状态 -> 转换为仿真兼容的数组
        关键映射：Sim J3 = -Real J3
        """
        if not self.robot.is_connected:
            return None
            
        obs = self.robot.get_observation()
        q = np.zeros(7) 
        
        q[0] = obs.get('joint_1.pos', 0) * HARDWARE_DIR['joint_1']
        q[1] = obs.get('joint_2.pos', 0) * HARDWARE_DIR['joint_2']
        q[2] = obs.get('joint_3.pos', 0) * HARDWARE_DIR['joint_3']
        q[3] = obs.get('joint_4.pos', 0) * HARDWARE_DIR['joint_4']
        q[4] = obs.get('joint_5.pos', 0) * HARDWARE_DIR['joint_5']
        q[5] = obs.get('joint_6.pos', 0) * HARDWARE_DIR['joint_6']
        
        # 映射夹爪: Real(0.0=Open, 1.0=Closed) -> Sim(0.04=Open, 0.0=Closed)
        g_norm = obs.get('gripper.pos', 0) 
        q[6] = (1.0 - g_norm) * 0.04
        
        return q

    def read_raw_dict(self):
        if not self.robot.is_connected: 
            return {}
        return self.robot.get_observation()

    def send_joints_from_sim(self, q_sim):
        if not self.robot.is_connected: return
        action = {}
        vals = {
            "joint_1": q_sim[0] * HARDWARE_DIR['joint_1'],
            "joint_2": q_sim[1] * HARDWARE_DIR['joint_2'],
            "joint_3": q_sim[2] * HARDWARE_DIR['joint_3'],
            "joint_4": q_sim[3] * HARDWARE_DIR['joint_4'],
            "joint_5": q_sim[4] * HARDWARE_DIR['joint_5'],
            "joint_6": q_sim[5] * HARDWARE_DIR['joint_6']
        }
        for k, v in vals.items():
            action[f"{k}.pos"] = np.clip(v, REAL_JOINT_LIMITS[k][0], REAL_JOINT_LIMITS[k][1])
            
        sim_g = np.clip(q_sim[6], 0.0, 0.04)
        g_val = 1.0 - (sim_g / 0.04)
        action['gripper.pos'] = np.clip(g_val, 0.0, 1.0)
        #print(action)
        self.robot.send_action(action)

    def send_raw_action(self, action_dict):
        if self.robot.is_connected: 
            self.robot.send_action(action_dict)

    def disconnect(self):
        if self.robot.is_connected:
            self.robot.disconnect()


# ==========================================
# 1. IK 解算器 (保持 LOCAL_WORLD_ALIGNED)
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
        # 仿真中的舒适姿态 (Sim坐标系：J3为负)
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
# 2. 6自由度仿真臂
# ==========================================
class SixDofArm:
    def __init__(self, urdf_path, mesh_dir):
        self.model, self.collision_model, self.visual_model = self._load_model(urdf_path, mesh_dir)
        self.data = self.model.createData()
        
        # [关键配置] 这里的限位是给 IK 用的，必须使用仿真坐标系
        # 但是，我们必须把范围限制在“真实机器能达到的范围内”
        # Real J3 [0, 3] -> Sim J3 [-3, 0]
        self.joint_limits = [
            [-3.0, 3.0],   # J1
            [-0.3, 3.0],   # J2
            [-3.0, 0.0],   # J3 (Sim坐标系)
            [-1.7, 1.2],   # J4
            [-0.4, 0.4],   # J5 (已收窄，匹配真机)
            [-2.0, 2.0],   # J6
            [0.0, 0.04],   # Gripper
        ]
        
        if self.model.existFrame("link4"):
            self.ik_frame_id = self.model.getFrameId("link4")
        else:
            self.ik_frame_id = self.model.getFrameId("link3")
            
        self.ik_solver = ThreeDofIKSolver(self.model, self.data, self.ik_frame_id, self.joint_limits[:3])
        
        # 初始化姿态 (Sim坐标系)
        self.q = pin.neutral(self.model)
        self.q[0] = 0.020
        self.q[1] = 1.671  
        self.q[2] = -0.670 # J3 (Sim坐标系)
        self.q[3] = -1.20
        self.q[4] = 0.0
        self.q[5] = 0.0
        
        self.in_zero_mode = False

        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy() 

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
        """ 标准复位 """
        self.q = pin.neutral(self.model)
        self.q[0] = 0.020
        self.q[1] = 1.671  
        self.q[2] = -0.670 # Sim坐标系
        self.q[3] = -1.20 
        self.q[4] = 0.0   
        self.q[5] = 0.0   
        self.q[6:] = 0.0  
        
        self.in_zero_mode = False
        self.ik_solver.q_ref_3dof = np.array([0.0, 1.5, -1.0])

        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy()
        logger.info("🔄 Standard Reset (J4=-1.20)")

    def reset_to_zero(self):
        """ 全关节归零 """
        self.q = np.zeros(self.model.nq) 
        self.in_zero_mode = True
        self.ik_solver.q_ref_3dof = np.array([0.0, 0.0, 0.0]) 

        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy()
        logger.info("⚠️ ALL JOINTS ZEROED")

    def set_state_from_hardware(self, q_real):
        """ SYNC 模式：q_real 已经是 SixDofRealArm 转换过的 Sim 坐标系数据 """
        n = min(len(self.q), len(q_real))
        self.q[:n] = q_real[:n]
        
        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy()
        
        self.ik_solver.q_ref_3dof = self.q[:3].copy()
        self.in_zero_mode = False

    def update(self, xyz_delta, manual_controls, dt=0.1):
        has_input = np.linalg.norm(xyz_delta) > 1e-6 or any(val != 0 for val in manual_controls.values())
        if has_input:
            self.in_zero_mode = False
            #if np.linalg.norm(self.ik_solver.q_ref_3dof) < 0.1:
            #     self.ik_solver.q_ref_3dof = np.array([0.0, 1.5, -1.0])

        # 1. 关节控制
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
                self.q[7] -= delta 
                self.q[7] = np.clip(self.q[7], -self.joint_limits[6][1], self.joint_limits[6][0])

        # 2. XYZ IK 解算

        # A 暂存旧的有效位置 (用于失败回退)
        old_safe_pos = self.valid_target_pos.copy()

        # B. 先应用用户输入
        self.target_pos += xyz_delta
        clamped_msg = ""
        
        # c. 计算“理想的”合规位置 (Shadow Target)，但不直接赋值
        ideal_pos = self.target_pos.copy()

        if not self.in_zero_mode:
            # 应用所有空间限制到 ideal_pos
            if ideal_pos[1] > MAX_Y: 
                ideal_pos[1] = MAX_Y
            if ideal_pos[2] < MIN_JOINT4_Z: 
                ideal_pos[2] = MIN_JOINT4_Z
            
            # 最小半径限制 (防奇点核心)
            xy_dist = np.linalg.norm(ideal_pos[:2])
            if xy_dist < MIN_RADIUS_XY:
                if xy_dist < 1e-6: 
                    ideal_pos[:2] = [0, -MIN_RADIUS_XY]
                else: 
                    ideal_pos[:2] *= (MIN_RADIUS_XY / xy_dist)
            # 最大半径限制
            dist = np.linalg.norm(ideal_pos)
            if dist > MAX_RADIUS:
                ideal_pos *= (MAX_RADIUS / dist)

        else:
            clamped_msg = "⚠️ Zero Mode"

        # D. 平滑修正：将 target_pos 慢慢拉向 ideal_pos
        # 即使处于违规区域，每帧最多只修正 2mm，避免瞬移
        SAFETY_SNAP_SPEED = 0.002  # 修正速度：2mm/帧 (约12cm/s)
        
        diff = ideal_pos - self.target_pos
        dist_err = np.linalg.norm(diff)
        
        if dist_err > 1e-6:
            clamped_msg = "🔒 SmoothClamp"
            # 如果偏差很大，则每帧只修正一点点
            if dist_err > SAFETY_SNAP_SPEED:
                self.target_pos += (diff / dist_err) * SAFETY_SNAP_SPEED
            else:
                # 如果偏差很小，直接吸附过去
                self.target_pos = ideal_pos        

        # # --- 动态调整 J1 的参考角度 ---
        # # 计算当前目标点的朝向 (Yaw)
        # curr_xy_dist = np.linalg.norm(self.target_pos[:2])
        # if curr_xy_dist > 0.01: # 只有离原点有一定距离时计算才有意义
        #     target_yaw = np.arctan2(self.target_pos[1], self.target_pos[0])
        #     # 告诉 IK：你的舒适姿态应该是正对着目标点，而不是死板地盯着 0 度
        #     self.ik_solver.q_ref_3dof[0] = target_yaw

        # E. IK 解算
        q_new, debug_msg, cond, success, err = self.ik_solver.solve(self.target_pos, self.q)
        
        if not success:
            if not self.in_zero_mode: 
                # 失败时直接回退到上一步的有效位置，就像撞墙一样停住。
                # 绝对不要乘 0.99，那会把你吸入奇点黑洞！
                self.target_pos = old_safe_pos.copy()
                debug_msg += " -> BLOCKED"
        else:
            self.q = q_new
            if err < 0.02:
                self.valid_target_pos = self.target_pos.copy()
                
        return debug_msg, cond, clamped_msg, success


# ==========================================
# 3. 仿真主循环
# ==========================================
class SixDofSim:
    def __init__(self, use_real_arm=False):
        pygame.init()
        pygame.joystick.init()
        self.js = None
        if pygame.joystick.get_count() > 0:
            self.js = pygame.joystick.Joystick(0)
            self.js.init()
            logger.info(f"🎮 Joystick: {self.js.get_name()}")
        
        self.arm = SixDofArm(URDF_PATH, MESH_DIR)
        
        self.real_arm = None
        if use_real_arm:
            if HAS_REAL_ARM_LIB:
                try:
                    self.real_arm = SixDofRealArm(REAL_ARM_PORT)
                    logger.info("✅ Real Robot Mode Activated")
                except Exception as e:
                    logger.error(f"❌ Failed to connect Real Arm: {e}")
                    logger.warning("⚠️ Fallback to Simulation Only")
            else:
                logger.warning("⚠️ Mode is 'real' but 'follower_mkarm' library is missing. Running in SIM mode.")
        else:
            logger.info("💻 Running in Simulation Only Mode (Safe)")
        
        self.viz = MeshcatVisualizer(self.arm.model, self.arm.collision_model, self.arm.visual_model)
        try:
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
        except: pass
        
        self._init_visuals()
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.x_press_start_time = None
        self.zero_reset_done = False
        self.mode_joint_ctrl = False;  #手柄直接控制joints值
        self.last_back_btn = 0

        self.test_target_joints = {}
        if self.real_arm: 
            self._sync_test_target_from_real()
        else: 
            self.test_gripper_pos = 0.0

        self.is_homing = False       # 是否正在自动归零中
        self.rb_safety_lock = False  # 是否处于安全锁定（等待松开RB）

    def _init_visuals(self):
        self.viz.viewer["target"].set_object(g.Sphere(0.04), g.MeshBasicMaterial(color=0xff0000, opacity=0.8))
        self.viz.viewer["workspace_outer"].set_object(g.Sphere(MAX_RADIUS), 
                                        g.MeshBasicMaterial(color=0xffffff, opacity=1, wireframe=True))
        cyl_geom = g.Cylinder(0.4, MIN_RADIUS_XY, MIN_RADIUS_XY)
        self.viz.viewer["workspace_inner"].set_object(cyl_geom, 
                                        g.MeshBasicMaterial(color=0xff0000, opacity=1, wireframe=False))
        self.viz.viewer["workspace_inner"].set_transform(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0.2],[0,0,0,1]]))

    def _sync_test_target_from_real(self):
        obs = self.real_arm.read_raw_dict()
        for i in range(1, 7):
            key = f"joint_{i}"
            self.test_target_joints[key] = obs.get(f"{key}.pos", 0.0)
        self.test_gripper_pos = obs.get("gripper.pos", 0.0)

    def _filter_stick(self, val):
        if abs(val) < 0.15: return 0.0
        return val

    def _get_inputs(self):
        pygame.event.pump()
        xyz_delta = np.zeros(3)
        manual = {'j4':0, 'j5':0, 'j6':0, 'gripper':0}
        
        # 获取物理 RB 键状态
        phys_rb_pressed = self.js.get_button(BTN_RB) == 1
        
        #处理 RB 安全锁
        # 如果处于锁定状态：强制认为 RB 没按，直到物理 RB 松开
        if self.rb_safety_lock:
            if not phys_rb_pressed:
                self.rb_safety_lock = False # 解锁
                logger.info("🔓 RB Released - Safety Lock Disengaged")
            final_rb_pressed = False
        else:
            final_rb_pressed = phys_rb_pressed

        if not self.js: 
            return xyz_delta, manual, False, False

        # 1. 模式切换 (Back键)
        back_click = False
        if self.js.get_button(BTN_BACK) and not self.last_back_btn: 
            back_click = True
        self.last_back_btn = self.js.get_button(BTN_BACK)

        # 2. 复位逻辑 (X键) - 仅在 IK 模式有效
        if  not self.mode_joint_ctrl:
            x_btn_state = self.js.get_button(BTN_X)
            if x_btn_state == 1: 
                if self.x_press_start_time is None:
                    self.x_press_start_time = time.time()
                    if not self.is_homing:
                        self.arm.reset() # 短按普通复位
                    # 短按期间，强制断开 RB，防止瞬移
                    final_rb_pressed = False
                else:
                    duration = time.time() - self.x_press_start_time
                    if duration > LONG_PRESS_TIME :
                        if not self.is_homing:
                            logger.info("🚀 Starting Smooth Homing to ZERO...")
                            self.is_homing = True # 开启归位模式
                        # 在归位过程中，强制允许发送指令 (忽略 RB 锁)
                        # 但归位结束后，会进入 safety_lock
                        final_rb_pressed = True
                    else:
                        # 长按未达到时间时，保持断开，等待触发
                        final_rb_pressed = False
                return np.zeros(3), manual, final_rb_pressed , back_click # 复位时不移动
            else: 
                self.x_press_start_time = None

        lx = self._filter_stick(self.js.get_axis(AXIS_LX))
        ly = self._filter_stick(self.js.get_axis(AXIS_LY))
        ry = self._filter_stick(self.js.get_axis(AXIS_RY))
        rx = self._filter_stick(self.js.get_axis(AXIS_RX))
        hat = self.js.get_hat(HAT_ID)

        xyz_delta[0] = CONTROL_DIR['IK_X'] * lx * TRANS_SPEED
        xyz_delta[1] = CONTROL_DIR['IK_Y'] * ly * TRANS_SPEED
        xyz_delta[2] = CONTROL_DIR['IK_Z'] * ry * TRANS_SPEED
        
        manual['j4'] = -hat[1] 
        manual['j5'] = -rx 
        manual['j6'] = -hat[0] 
        
        
        lt_val = (self.js.get_axis(AXIS_LT) + 1) / 2 
        rt_val = (self.js.get_axis(AXIS_RT) + 1) / 2     
        if rt_val > 0.1: manual['gripper'] = 1 
        elif lt_val > 0.1: manual['gripper'] = -1
        
        #rb_pressed = self.js.get_button(BTN_RB) == 1
            
        return xyz_delta, manual, final_rb_pressed, back_click

    def run(self):
        logger.info("🚀 Simulation Loop Started")
        force_flush_log() # [关键] 启动时立即写入硬盘，防止开局崩溃无日志
        
        log_counter = 0
        HOMING_SPEED = 0.005 # 归位速度 (弧度/帧)，约 0.3 rad/s，平滑缓慢
        
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: 
                        self.running = False
                
                xyz_delta, manual_ctrl, rb_pressed , back_click = self._get_inputs()
                sim_mode_str = "💻 SIM ONLY"

                if back_click: #已经切换到直控joints模式,sync the real arm joints to sim
                    self.mode_joint_ctrl = not self.mode_joint_ctrl
                    if self.real_arm: 
                        self._sync_test_target_from_real()
                
                # 优先处理 Homing 归位模式
                if self.is_homing:
                    sim_mode_str = "♻️ HOMING..."
                    # 1. 计算插值: 让每个关节缓慢趋向 0
                    max_diff = 0.0
                    for i in range(len(self.arm.q)):
                        diff = 0.0 - self.arm.q[i]
                        step = np.sign(diff) * min(abs(diff), HOMING_SPEED)
                        self.arm.q[i] += step
                        max_diff = max(max_diff, abs(diff))
                    
                    # 2. 更新 FK (保证 visualizer 和 target_pos 同步)
                    pin.framesForwardKinematics(self.arm.model, self.arm.data, self.arm.q)
                    self.arm.target_pos = self.arm.data.oMf[self.arm.ik_frame_id].translation.copy()
                    
                    # 3. 发送给真机 (rb_pressed 在 Homing 时被强制为 True)
                    if self.real_arm and rb_pressed:
                        self.real_arm.send_joints_from_sim(self.arm.q)
                    
                    # 4. 判断是否到达 (允许 0.01 弧度误差)
                    if max_diff < 0.01:
                        self.is_homing = False
                        self.arm.reset_to_zero() # 最终对齐
                        self.rb_safety_lock = True # [关键] 开启安全锁，防止 RB 误触
                        logger.info("✅ Homing Complete. Safety Lock Engaged (Release RB).")
                    
                    # Homing 期间跳过后续逻辑
                    info_str = f"{sim_mode_str} | Dist: {max_diff:.3f}"

                elif self.mode_joint_ctrl :
                    if not self.real_arm:
                         print("REAL ARM NOT READY, CAN NOT STAY IN CTRL JONTS MODE")
                         sim_mode_str = "⚠️ REAL ARM NOT READY, CAN NOT STAY IN CTRL JONTS MODE"
                         self.mode_joint_ctrl = False
                    else:
                        sim_mode_str = "🛠️ CTRL REAL JOINTS"
                        lx = self._filter_stick(self.js.get_axis(AXIS_LX))
                        ly = self._filter_stick(self.js.get_axis(AXIS_LY))
                        rx = self._filter_stick(self.js.get_axis(AXIS_RX))
                        ry = self._filter_stick(self.js.get_axis(AXIS_RY))
                        hat = self.js.get_hat(HAT_ID)

                        self.test_target_joints['joint_1'] += lx     * JOINT_CTRL_SPEED_J1_J3 * CONTROL_DIR['CTRL_J1']
                        self.test_target_joints['joint_2'] += ly     * JOINT_CTRL_SPEED_J1_J3 * CONTROL_DIR['CTRL_J2']
                        self.test_target_joints['joint_3'] += ry     * JOINT_CTRL_SPEED_J1_J3 * CONTROL_DIR['CTRL_J3']
                        self.test_target_joints['joint_4'] += hat[1] * JOINT_CTRL_SPEED_J4_J6 * CONTROL_DIR['CTRL_J4']
                        self.test_target_joints['joint_5'] += rx     * JOINT_CTRL_SPEED_J4_J6 * CONTROL_DIR['CTRL_J5']
                        self.test_target_joints['joint_6'] += hat[0] * JOINT_CTRL_SPEED_J4_J6 * CONTROL_DIR['CTRL_J6']
                        
                        if manual_ctrl['gripper'] > 0: self.test_gripper_pos += GRIPPER_SPEED
                        elif manual_ctrl['gripper'] < 0: self.test_gripper_pos -= GRIPPER_SPEED
                        self.test_gripper_pos = np.clip(self.test_gripper_pos, 0.0, 1.0)
                        
                        # 真机限位检查
                        for k in REAL_JOINT_LIMITS:
                            current_val = self.test_target_joints.get(k, 0.0)
                            min_val, max_val = REAL_JOINT_LIMITS[k]
                            if k == "joint_2": # 重点监控 J2
                                if current_val <= min_val + 0.01: limit_alert = "⚠️ J2 MIN!"
                                elif current_val >= max_val - 0.01: limit_alert = "⚠️ J2 MAX!"
                            self.test_target_joints[k] = np.clip(current_val, min_val, max_val)

                        if rb_pressed:
                            act = {f"{k}.pos": v for k,v in self.test_target_joints.items()}
                            act['gripper.pos'] = self.test_gripper_pos
                            self.real_arm.send_raw_action(act)
                            sim_mode_str = "🛠️ SEND"
                        else: sim_mode_str = "🛠️ HOLD RB"
                        
                        q_real = self.real_arm.read_joints()
                        if q_real is not None: 
                            self.arm.set_state_from_hardware(q_real)
                        
                        info_str = (f"{sim_mode_str} | {debug_msg} {clamp_msg} | "
                            f"J:[{self.arm.q[0]:.2f}, {self.arm.q[1]:.2f}, {self.arm.q[2]:.2f}, "
                            f"{self.arm.q[3]:.2f}, {self.arm.q[4]:.2f}, {self.arm.q[5]:.2f}]")

                else :
                    if self.real_arm:  #通过IK计算，仿真，再到真机
                        if rb_pressed:
                            # [CONTROL 模式]
                            debug_msg, cond, clamp_msg, success = self.arm.update(xyz_delta, manual_ctrl)
                            if success:
                                self.real_arm.send_joints_from_sim(self.arm.q)
                                sim_mode_str = "🎮 CTL -> REAL"
                            else:
                                sim_mode_str = "⛔ CTL BLOCKED (IK Err)"
                        else:
                            # [SYNC 模式]
                            # 如果此时处于 Safety Lock 状态，rb_pressed 会被强制为 False
                            # 代码会正确地进入这里，读取真机数据（此时真机应该已经在 0 位了）
                            q_real = self.real_arm.read_joints()
                            if q_real is not None:
                                self.arm.set_state_from_hardware(q_real)
                                debug_msg, cond, clamp_msg = "Syncing", 0.0, ""
                                sim_mode_str = "👁️ SYNC <- REAL"
                                if self.rb_safety_lock: 
                                    sim_mode_str = "🔒 RELEASE RB"
                            else:
                                #debug_msg, cond, clamp_msg, success = self.arm.update(xyz_delta, manual_ctrl)
                                sim_mode_str = "⚠️ READ FAIL"
                        info_str = (f"{sim_mode_str} | {debug_msg} {clamp_msg} | "
                            f"Tgt:[{self.arm.target_pos[0]:.3f}, {self.arm.target_pos[1]:.3f}, {self.arm.target_pos[2]:.3f}] | "
                            f"J:[{self.arm.q[0]:.2f}, {self.arm.q[1]:.2f}, {self.arm.q[2]:.2f}, "
                            f"{self.arm.q[3]:.2f}, {self.arm.q[4]:.2f}, {self.arm.q[5]:.2f}]")
                    
                    else: #sim only
                        debug_msg, cond, clamp_msg, success = self.arm.update(xyz_delta, manual_ctrl)
                        info_str = (f"{sim_mode_str} | {debug_msg} {clamp_msg} | "
                            f"Tgt:[{self.arm.target_pos[0]:.3f}, {self.arm.target_pos[1]:.3f}, {self.arm.target_pos[2]:.3f}] | "
                            f"J:[{self.arm.q[0]:.2f}, {self.arm.q[1]:.2f}, {self.arm.q[2]:.2f}, "
                            f"{self.arm.q[3]:.2f}, {self.arm.q[4]:.2f}, {self.arm.q[5]:.2f}]")

                # --- 可视化 ---
                self.viz.display(self.arm.q)
                self.viz.viewer["target"].set_transform(pin.SE3(np.eye(3), self.arm.target_pos).homogeneous)
                
                # 颜色指示状态
                target_color = 0xff0000 # 红
                if self.is_homing: 
                    target_color = 0xffff00 # 黄色 (归位中)
                elif self.rb_safety_lock: 
                    target_color = 0xffa500 # 橙色 (等待解锁)
                elif rb_pressed and self.real_arm: 
                    target_color = 0x00ff00 # 绿色 (正常控制)
                elif self.real_arm: 
                    target_color = 0x0000ff # 蓝色 (同步)
                
                self.viz.viewer["target"].set_object(g.Sphere(0.04), g.MeshBasicMaterial(color=target_color, opacity=0.8))    
                
                print(info_str, end='\r')
                
                log_counter += 1
                # 如果 IK 发散或者每隔 20 帧，记录一次日志
                if log_counter % 20 == 0 or "Diverged" in debug_msg:
                    logger.info(info_str)
                    force_flush_log() # [关键] 每次写入日志后，强制刷新到硬盘
                    
                self.clock.tick(FREQ)
                
        except KeyboardInterrupt:
            logger.info("⚠️ Interrupted by user (Ctrl+C)")
            force_flush_log()
        except Exception as e:
            logger.critical(f"❌ Runtime Error: {e}", exc_info=True) # 使用 critical 级别记录崩溃
            force_flush_log()
        finally:
            if self.real_arm:
                logger.info("Disconnecting real arm...")
                self.real_arm.disconnect()
            pygame.quit()
            logger.info("🛑 Simulation Ended")
            force_flush_log()
            logging.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6-DoF Arm Teleoperation & Simulation")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="sim", 
        choices=["sim", "real"], 
        help="Operation mode: 'sim' (Simulation only) or 'real' (Simulation + Real Robot)"
    )
    
    args = parser.parse_args()
    use_real = (args.mode == "real")
    logger.info(f"Arguments: mode={args.mode} -> use_real_arm={use_real}")
    
    try:
        sim = SixDofSim(use_real_arm=use_real)
        sim.run()
    except Exception as e:
        logger.critical(f"🔥 Fatal Error: {e}", exc_info=True)