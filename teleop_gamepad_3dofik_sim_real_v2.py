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

# ==========================================
# 1. 全局配置
# ==========================================
URDF_PATH = "hardware/urdf/urdf/dk2.SLDASM.urdf"
MESH_DIR = "hardware/urdf" 
FREQ = 60 
TRANS_SPEED = 0.005   # IK模式: 末端移动速度 (XYZ)
JOINT_SPEED = 0.05    # IK模式: 关节旋转速度 (J4-J6)
GRIPPER_SPEED = 0.02  # 夹爪速度

# --- [新增] 关节测试模式的控制速度 ---
TEST_SPEED_J1_J3 = 0.01
TEST_SPEED_J4_J6 = 0.01

LONG_PRESS_TIME = 2.0 
REAL_ARM_PORT = "/dev/ttyACM0" 

# 空间限制参数 (IK用)
MAX_RADIUS = 0.5      
MIN_RADIUS_XY = 0.05  
MIN_JOINT4_Z = 0.0          
MAX_Y = 0  

# 真实机械臂的物理限位
REAL_JOINT_LIMITS = {
    "joint_1": [-3.0, 3.0],
    "joint_2": [-0.3, 3.0],
    "joint_3": [0.0, 3.0],   # Real J3 是正值区间
    "joint_4": [-1.7, 1.2],
    "joint_5": [-0.4, 0.4],  
    "joint_6": [-2.0, 2.0]
}

# 手柄按键映射 (Xbox Controller)
BTN_A = 0
BTN_B = 1
BTN_X = 2  # Reset
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5 # [Deadman Switch]
BTN_BACK = 6 # [新增] 切换模式按键 (View/Back)
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
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_path = os.path.join(LOG_DIR, log_filename)

logger = logging.getLogger("MKArmLogger")
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def force_flush_log():
    file_handler.flush()
    try: os.fsync(file_handler.stream.fileno())
    except: pass

logger.info(f"📝 Log file: {log_path}")
force_flush_log()

# ==========================================
# 2. 真实机械臂接口类
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
        """ 读取真实状态 -> 转换为 Sim 状态 (Sim J3 = -Real J3) """
        if not self.robot.is_connected: return None
        obs = self.robot.get_observation()
        q = np.zeros(7) 
        q[0] = obs.get('joint_1.pos', 0)
        q[1] = obs.get('joint_2.pos', 0)
        q[2] = -obs.get('joint_3.pos', 0) # [映射] Real(+) -> Sim(-)
        q[3] = obs.get('joint_4.pos', 0)
        q[4] = obs.get('joint_5.pos', 0)
        q[5] = obs.get('joint_6.pos', 0)
        g_norm = obs.get('gripper.pos', 0) 
        q[6] = (1.0 - g_norm) * 0.04
        return q

    def read_raw_dict(self):
        """ 读取原始字典（用于测试模式的积分初始值） """
        if not self.robot.is_connected: return {}
        return self.robot.get_observation()

    def send_joints_from_sim(self, q_sim):
        """ IK模式用: Sim角度 -> Real角度 (处理 J3 符号) """
        if not self.robot.is_connected: return
        action = {}
        # 1. 映射 Sim -> Real
        real_vals = [q_sim[0], q_sim[1], -q_sim[2], q_sim[3], q_sim[4], q_sim[5]]
        
        # 2. 安全限位 + 构建字典
        keys = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        for i, key in enumerate(keys):
            action[f"{key}.pos"] = np.clip(real_vals[i], REAL_JOINT_LIMITS[key][0], REAL_JOINT_LIMITS[key][1])
            
        # 3. 夹爪
        sim_g = np.clip(q_sim[6], 0.0, 0.04)
        g_val = 1.0 - (sim_g / 0.04)
        action['gripper.pos'] = np.clip(g_val, 0.0, 1.0)
        
        self.robot.send_action(action)

    def send_raw_action(self, action_dict):
        """ 测试模式用: 直接发送构建好的字典 """
        if self.robot.is_connected:
            self.robot.send_action(action_dict)

    def disconnect(self):
        if self.robot.is_connected:
            self.robot.disconnect()

# ==========================================
# 3. IK 解算器 & 仿真臂类
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
        success = False
        final_err = 0.0
        cond = 0.0
        
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
            debug_info = f"⛔ Diverged ({final_err*100:.0f}cm)"
            success = False
        elif debug_info == "":
            debug_info = "✅ Reached"
            success = True
        return q, debug_info, cond, success, final_err

class SixDofArm:
    def __init__(self, urdf_path, mesh_dir):
        self.model, self.collision_model, self.visual_model = self._load_model(urdf_path, mesh_dir)
        self.data = self.model.createData()
        # IK 限位 (Sim坐标系)
        self.joint_limits = [
            [-3.0, 3.0], [-0.3, 3.0], [-3.0, 0.0], # J1-J3
            [-1.7, 1.2], [-0.4, 0.4], [-2.0, 2.0], # J4-J6
            [0.0, 0.04] # Gripper
        ]
        if self.model.existFrame("link4"): self.ik_frame_id = self.model.getFrameId("link4")
        else: self.ik_frame_id = self.model.getFrameId("link3")
            
        self.ik_solver = ThreeDofIKSolver(self.model, self.data, self.ik_frame_id, self.joint_limits[:3])
        self.reset()

    def _load_model(self, urdf_path, mesh_dir):
        abs_urdf_path = os.path.abspath(urdf_path)
        abs_mesh_dir = os.path.abspath(mesh_dir)
        meshes_folder_abs = os.path.join(abs_mesh_dir, "meshes")
        with open(abs_urdf_path, 'r') as f: urdf_content = f.read()
        urdf_content = urdf_content.replace('filename="package://dk2.SLDASM/meshes/', f'filename="{meshes_folder_abs}/')
        urdf_content = urdf_content.replace('filename="../meshes/', f'filename="{meshes_folder_abs}/')
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf', delete=False) as tmp:
            tmp.write(urdf_content)
            tmp_urdf_path = tmp.name
        model = pin.buildModelFromXML(urdf_content)
        visual_model = pin.buildGeomFromUrdf(model, tmp_urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_dir)
        collision_model = pin.buildGeomFromUrdf(model, tmp_urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)
        os.remove(tmp_urdf_path)
        return model, collision_model, visual_model

    def reset(self):
        self.q = pin.neutral(self.model)
        self.q[0:6] = [0.020, 1.671, -0.670, -1.20, 0.0, 0.0]
        self.in_zero_mode = False
        self._update_tgt()

    def reset_to_zero(self):
        self.q = np.zeros(self.model.nq)
        self.in_zero_mode = True
        self.ik_solver.q_ref_3dof = np.array([0.0, 0.0, 0.0])
        self._update_tgt()

    def set_state_from_hardware(self, q_real_sim_frame):
        """ 仅用于可视化同步 """
        n = min(len(self.q), len(q_real_sim_frame))
        self.q[:n] = q_real_sim_frame[:n]
        self._update_tgt()
        if not self.in_zero_mode: self.ik_solver.q_ref_3dof = self.q[:3].copy()

    def _update_tgt(self):
        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy()

    def update_ik(self, xyz_delta, manual_controls):
        # ... (IK逻辑保持不变，为节省篇幅略去细节，逻辑同原文件) ...
        # 1. 关节 J4-J6 更新
        if manual_controls['j4']: self.q[3] = np.clip(self.q[3] + manual_controls['j4']*JOINT_SPEED, -1.7, 1.2)
        if manual_controls['j5']: self.q[4] = np.clip(self.q[4] + manual_controls['j5']*JOINT_SPEED, -0.4, 0.4)
        if manual_controls['j6']: self.q[5] = np.clip(self.q[5] + manual_controls['j6']*JOINT_SPEED, -2.0, 2.0)
        if manual_controls['gripper']: 
             self.q[6] = np.clip(self.q[6] + manual_controls['gripper']*GRIPPER_SPEED, 0.0, 0.04)
             if len(self.q)>7: self.q[7] = -self.q[6] # 模拟双指

        # 2. XYZ 更新
        self.target_pos += xyz_delta
        if not self.in_zero_mode:
            if self.target_pos[1] > MAX_Y: self.target_pos[1] = MAX_Y
            if self.target_pos[2] < MIN_JOINT4_Z: self.target_pos[2] = MIN_JOINT4_Z
            xy_dist = np.linalg.norm(self.target_pos[:2])
            if xy_dist < MIN_RADIUS_XY: self.target_pos[:2] = [0, -MIN_RADIUS_XY] if xy_dist<1e-6 else self.target_pos[:2]*(MIN_RADIUS_XY/xy_dist)
            if np.linalg.norm(self.target_pos) > MAX_RADIUS: self.target_pos *= (MAX_RADIUS/np.linalg.norm(self.target_pos))
        
        q_new, debug_msg, cond, success, err = self.ik_solver.solve(self.target_pos, self.q)
        if success: 
            self.q = q_new
            if err < 0.02: self.valid_target_pos = self.target_pos.copy()
        else:
            if not self.in_zero_mode: 
                self.target_pos = self.valid_target_pos.copy() * 0.99
                self.valid_target_pos = self.target_pos.copy()
        
        return debug_msg, cond, "", success

# ==========================================
# 4. 仿真主程序 (包含状态机)
# ==========================================
class SixDofSim:
    def __init__(self, use_real_arm=False):
        pygame.init()
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        if self.js: self.js.init()
        
        self.arm = SixDofArm(URDF_PATH, MESH_DIR)
        
        self.real_arm = None
        self.use_real = use_real_arm
        if use_real_arm and HAS_REAL_ARM_LIB:
            try:
                self.real_arm = SixDofRealArm(REAL_ARM_PORT)
            except Exception as e:
                logger.error(f"❌ Real Arm Connect Fail: {e}")
        
        # 可视化
        self.viz = MeshcatVisualizer(self.arm.model, self.arm.collision_model, self.arm.visual_model)
        try: self.viz.initViewer(open=True); self.viz.loadViewerModel()
        except: pass
        self._init_visuals()
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # --- 模式控制 ---
        self.mode_joint_test = False # False=IK模式, True=关节测试模式
        self.last_back_btn = 0
        
        # 关节测试模式下的积分状态
        self.test_target_joints = {} 
        if self.real_arm:
            # 初始化为当前真机状态
            obs = self.real_arm.read_raw_dict()
            for k in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
                self.test_target_joints[k] = obs.get(f"{k}.pos", 0.0)
            self.test_gripper_pos = obs.get("gripper.pos", 0.0)
        else:
            # 这里的初始值只是为了不报错，没连真机进不了这个模式
            for k in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
                self.test_target_joints[k] = 0.0
            self.test_gripper_pos = 0.0

    def _init_visuals(self):
        self.viz.viewer["target"].set_object(g.Sphere(0.04), g.MeshBasicMaterial(color=0xff0000, opacity=0.8))

    def _filter_stick(self, val, dz=0.15):
        return 0.0 if abs(val) < dz else val

    def _get_inputs(self):
        pygame.event.pump()
        if not self.js: return None

        # --- 1. 检测模式切换 (Back键) ---
        back_btn = self.js.get_button(BTN_BACK)
        if back_btn == 1 and self.last_back_btn == 0:
            self.mode_joint_test = not self.mode_joint_test
            mode_str = "🛠️ JOINT TEST MODE" if self.mode_joint_test else "🎮 IK CONTROL MODE"
            logger.info(f"🔀 Switched to: {mode_str}")
            # 切换模式时，重新同步一次真机位置作为起点
            if self.real_arm and self.real_arm.robot.is_connected:
                obs = self.real_arm.read_raw_dict()
                for k in self.test_target_joints.keys():
                    self.test_target_joints[k] = obs.get(f"{k}.pos", 0.0)
                self.test_gripper_pos = obs.get("gripper.pos", 0.0)
        self.last_back_btn = back_btn

        # --- 2. 通用读取 ---
        inputs = {}
        inputs['lx'] = self._filter_stick(self.js.get_axis(AXIS_LX))
        inputs['ly'] = self._filter_stick(self.js.get_axis(AXIS_LY))
        inputs['rx'] = self._filter_stick(self.js.get_axis(AXIS_RX))
        inputs['ry'] = self._filter_stick(self.js.get_axis(AXIS_RY))
        inputs['hat'] = self.js.get_hat(HAT_ID)
        inputs['lt'] = (self.js.get_axis(AXIS_LT) + 1) / 2
        inputs['rt'] = (self.js.get_axis(AXIS_RT) + 1) / 2
        inputs['rb_pressed'] = self.js.get_button(BTN_RB) == 1
        inputs['x_btn'] = self.js.get_button(BTN_X)
        
        return inputs

    def run(self):
        logger.info("🚀 Sim Started. Press [BACK] to toggle modes.")
        force_flush_log()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
            
            inp = self._get_inputs()
            status_line = ""
            
            # ===============================
            # 模式 A: 🛠️ 关节测试模式 (Direct Joint Control)
            # ===============================
            if self.mode_joint_test:
                if not self.real_arm:
                    status_line = "⚠️ No Real Arm connected for Test Mode"
                else:
                    # 1. 计算增量 (完全复用 ctrl_joints.py 的逻辑)
                    # LX -> J1 (左右)
                    self.test_target_joints["joint_1"] += inp['lx'] * TEST_SPEED_J1_J3
                    # LY -> J2 (前后)
                    self.test_target_joints["joint_2"] += inp['ly'] * TEST_SPEED_J1_J3
                    # RY -> J3 (前后)
                    self.test_target_joints["joint_3"] += inp['ry'] * TEST_SPEED_J1_J3
                    # DPad Y -> J4
                    self.test_target_joints["joint_4"] += -inp['hat'][1] * TEST_SPEED_J4_J6
                    # RX -> J5 (左右)
                    self.test_target_joints["joint_5"] += -inp['rx'] * TEST_SPEED_J4_J6
                    # DPad X -> J6
                    self.test_target_joints["joint_6"] += -inp['hat'][0] * TEST_SPEED_J4_J6
                    
                    # Gripper
                    if inp['rt'] > 0.1: self.test_gripper_pos += GRIPPER_SPEED
                    elif inp['lt'] > 0.1: self.test_gripper_pos -= GRIPPER_SPEED
                    self.test_gripper_pos = np.clip(self.test_gripper_pos, 0.0, 1.0)
                    
                    # 2. 安全限位
                    for k, v in self.test_target_joints.items():
                        lim = REAL_JOINT_LIMITS[k]
                        self.test_target_joints[k] = np.clip(v, lim[0], lim[1])

                    # 3. 发送指令 (仅当按住 RB 死人开关时)
                    if inp['rb_pressed']:
                        action = {f"{k}.pos": v for k,v in self.test_target_joints.items()}
                        action["gripper.pos"] = self.test_gripper_pos
                        self.real_arm.send_raw_action(action)
                        status_line = "🛠️ TEST: SENDING"
                    else:
                        status_line = "🛠️ TEST: HOLD RB TO MOVE"

                    # 4. [关键] 读取真机状态更新仿真 -> 验证闭环
                    # 即使我在手动控制，我也想看 read_joints 是否正确映射了 Sim 里的样子
                    q_real = self.real_arm.read_joints()
                    if q_real is not None:
                        self.arm.set_state_from_hardware(q_real)

            # ===============================
            # 模式 B: 🎮 IK 控制模式 (IK Control)
            # ===============================
            else:
                # 1. 解析 IK 输入
                xyz_delta = np.zeros(3)
                xyz_delta[0] = -inp['lx'] * TRANS_SPEED
                xyz_delta[1] = inp['ly'] * TRANS_SPEED
                xyz_delta[2] = -inp['ry'] * TRANS_SPEED
                
                manual = {'j4': -inp['hat'][1], 'j5': -inp['rx'], 'j6': -inp['hat'][0], 'gripper': 0}
                if inp['rt'] > 0.1: manual['gripper'] = 1
                elif inp['lt'] > 0.1: manual['gripper'] = -1
                
                # 2. 运行 IK 状态机
                if self.real_arm:
                    if inp['rb_pressed']:
                        # [Control] Sim算IK -> 发给真机
                        debug, _, _, success = self.arm.update_ik(xyz_delta, manual)
                        if success: 
                            self.real_arm.send_joints_from_sim(self.arm.q)
                            status_line = f"🎮 IK->REAL | {debug}"
                        else:
                            status_line = f"⛔ IK FAIL | {debug}"
                    else:
                        # [Sync] 读取真机 -> 更新Sim
                        q_real = self.real_arm.read_joints()
                        if q_real is not None:
                            self.arm.set_state_from_hardware(q_real)
                            status_line = "👁️ SYNC <- REAL"
                else:
                    # [Sim Only]
                    debug, _, _, _ = self.arm.update_ik(xyz_delta, manual)
                    status_line = f"💻 SIM | {debug}"

            # --- 可视化与打印 ---
            self.viz.display(self.arm.q)
            # 更新目标球位置
            self.viz.viewer["target"].set_transform(pin.SE3(np.eye(3), self.arm.target_pos).homogeneous)
            # 目标球颜色状态
            color = 0x00ff00 if inp['rb_pressed'] and self.real_arm else (0x0000ff if self.real_arm else 0xff0000)
            self.viz.viewer["target"].set_object(g.Sphere(0.04), g.MeshBasicMaterial(color=color, opacity=0.8))

            info = (f"{status_line} | "
                    f"J_Real(SimFrame):[{self.arm.q[0]:.2f}, {self.arm.q[1]:.2f}, {self.arm.q[2]:.2f}]")
            print(info, end='\r')
            self.clock.tick(FREQ)

        if self.real_arm: self.real_arm.disconnect()
        pygame.quit()
        logger.info("🛑 End")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sim", choices=["sim", "real"])
    args = parser.parse_args()
    try:
        SixDofSim(use_real_arm=(args.mode=="real")).run()
    except Exception as e:
        logger.critical(f"🔥 Error: {e}", exc_info=True)