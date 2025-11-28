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

# --- 1. 配置 ---
URDF_PATH = "hardware/urdf/urdf/dk2.SLDASM.urdf"
MESH_DIR = "hardware/urdf" 
FREQ = 60 
TRANS_SPEED = 0.005  # 移动速度

# 手柄按键映射 (Xbox Controller: 0=A, 1=B, 2=X, 3=Y)
BTN_RESET = 2 

# 关节限位
JOINT_LIMITS = [
    [-3.0, 3.0],   # J1
    [-0.3, 3.0],   # J2
    [-3.0, 0.0],   # J3
]

# --- 空间限制参数 ---
MAX_RADIUS = 0.5      
MIN_RADIUS_XY = 0.05  
MIN_Z = 0.0            
MAX_Y = 0           

# --- 日志设置 ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def filter_stick(val_1, val_2, deadzone=0.15, snap_ratio=0.4):
    if abs(val_1) < deadzone: val_1 = 0
    if abs(val_2) < deadzone: val_2 = 0
    if val_1 != 0 and val_2 != 0:
        abs_1, abs_2 = abs(val_1), abs(val_2)
        if abs_2 < abs_1 * snap_ratio: val_2 = 0
        elif abs_1 < abs_2 * snap_ratio: val_1 = 0
    return val_1, val_2

def get_xyz_input(joystick):
    pygame.event.pump()
    v_pos = np.zeros(3)
    lx, ly = filter_stick(joystick.get_axis(0), joystick.get_axis(1))
    rx, ry = filter_stick(joystick.get_axis(3), joystick.get_axis(4))
    v_pos[0] = -lx * TRANS_SPEED   
    v_pos[1] = ly * TRANS_SPEED  
    v_pos[2] = -ry * TRANS_SPEED  
    return v_pos

class SimJoint4Only:
    def __init__(self, urdf_path, mesh_dir):
        abs_urdf_path = os.path.abspath(urdf_path)
        abs_mesh_dir = os.path.abspath(mesh_dir)
        meshes_folder_abs = os.path.join(abs_mesh_dir, "meshes")
        with open(abs_urdf_path, 'r') as f: urdf_content = f.read()
        urdf_content = urdf_content.replace('filename="package://dk2.SLDASM/meshes/', f'filename="{meshes_folder_abs}/')
        urdf_content = urdf_content.replace('filename="../meshes/', f'filename="{meshes_folder_abs}/')
        
        self.model = pin.buildModelFromXML(urdf_content)
        self.data = self.model.createData()
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf', delete=False) as tmp:
            tmp.write(urdf_content)
            tmp_urdf_path = tmp.name
        try:
            self.visual_model = pin.buildGeomFromUrdf(self.model, tmp_urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_dir)
            self.collision_model = pin.buildGeomFromUrdf(self.model, tmp_urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)
        finally:
            os.remove(tmp_urdf_path)
        
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        
        if self.model.existFrame("link4"):
            self.frame_id = self.model.getFrameId("link4")
        else:
            self.frame_id = self.model.getFrameId("link3")
            
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

    def start_viewer(self):
        try:
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
            logger.info("🌐 浏览器仿真已启动！")
        except: pass

    # --- [核心修改] 带安全回滚机制的 IK ---
    def solve_pos_ik(self, target_pos, q_current, dt=0.1):
        # 1. 备份当前状态 (安全存档)
        q_backup = q_current.copy()
        q = q_current.copy()
        
        debug_info = "" 
        cond = 1.0
        final_err = 0.0
        
        q_ref = np.array([0.0, 1.5, -1.0, 0.0, 0.0, 0.0]) 
        w_bias = 0.05 
        MAX_ITER = 15
        
        success = False 

        for i in range(MAX_ITER): 
            pin.framesForwardKinematics(self.model, self.data, q)
            current_pos = self.data.oMf[self.frame_id].translation
            
            err = target_pos - current_pos
            final_err = np.linalg.norm(err) 
            
            if final_err < 1e-3: 
                success = True
                debug_info = "✅ Reached"
                break
            
            J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_sub = J[:3, :3]
            
            cond = np.linalg.cond(J_sub)
            
            # 平滑阻尼：根据条件数动态调整
            damp = 1e-3 + 0.001 * (max(0, cond - 30))**2
            damp = min(damp, 0.1) # 上限 0.1

            H = J_sub.dot(J_sub.T) + damp * np.eye(3)
            
            v = J_sub.T.dot(np.linalg.solve(H, err))
            bias_force = w_bias * (q_ref[:3] - q[:3])
            v += bias_force * 0.1 

            v = np.clip(v, -0.5, 0.5) 
            
            q[:3] += v * dt
            
            for k in range(3):
                q[k] = max(JOINT_LIMITS[k][0], min(q[k], JOINT_LIMITS[k][1]))
        
        # --- 🛡️ 熔断保护 ---
        if final_err > 0.05: 
            q = q_backup
            debug_info = f"⛔ IK Diverged (Err:{final_err*100:.1f}cm) -> RESET"
            if cond > 50: debug_info += " [Singular]"
        elif debug_info == "":
            debug_info = "✅ Reached"

        return q, debug_info, cond

def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        logger.info(f"🎮 手柄就绪: {js.get_name()}")
    else:
        js = None 
        logger.warning("❌ 未检测到手柄，无法控制")

    sim = SimJoint4Only(URDF_PATH, MESH_DIR)
    sim.start_viewer()
    
    q = pin.neutral(sim.model)
    q[0] = 0.020
    q[1] = 1.671  
    q[2] = -0.670 
    
    # 1. 备份初始关节状态 (用于复位)
    q_init = q.copy()
    
    sim.viz.display(q)
    
    # 2. 计算初始实际位置 (确保复位点 100% 可达)
    pin.framesForwardKinematics(sim.model, sim.data, q)
    start_pos = sim.data.oMf[sim.frame_id].translation.copy()
    
    # 3. 将目标点对齐到实际位置
    target_pos = start_pos.copy()
    
    logger.info(f"📍 初始目标已对齐: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    logger.info(f"ℹ️  按手柄 'X' 键 (Button {BTN_RESET}) 可强制复位")
    
    sim.viz.viewer["target"].set_object(g.Sphere(0.04), g.MeshBasicMaterial(color=0xff0000, opacity=0.8))

    sphere_geom = g.Sphere(MAX_RADIUS)
    sphere_mat = g.MeshBasicMaterial(color=0xffffff, opacity=0.1, wireframe=False) 
    sim.viz.viewer["workspace_outer"].set_object(sphere_geom, sphere_mat)
    
    cyl_geom = g.Cylinder(0.4, MIN_RADIUS_XY, MIN_RADIUS_XY)
    cyl_mat = g.MeshBasicMaterial(color=0xff0000, opacity=0.3, wireframe=False)
    sim.viz.viewer["workspace_inner"].set_object(cyl_geom, cyl_mat)
    rot_x_90 = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0.2], 
        [0, 0, 0, 1]
    ])
    sim.viz.viewer["workspace_inner"].set_transform(rot_x_90)


    clock = pygame.time.Clock()
    running = True
    
    logger.info(f"🚀 仿真开始 | 限制: Y<=0, R_inner>={MIN_RADIUS_XY}, R_outer<={MAX_RADIUS}")
    
    log_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # --- 4. 监听重置键 (X) ---
        # 如果按下，强制将一切状态回滚到 start_pos
        if js and js.get_button(BTN_RESET):
            logger.info("🔄 RESET TRIGGERED") # 打印日志确认触发
            q = q_init.copy()                 # 关节回滚
            target_pos = start_pos.copy()     # 目标回滚
            # 这里不需要 continue，让它接着渲染一帧复位后的状态，体验更顺滑

        if js: v_pos = get_xyz_input(js)
        else: v_pos = np.zeros(3)

        target_pos += v_pos
        
        old_pos = target_pos.copy()
        
        if target_pos[1] > MAX_Y: target_pos[1] = MAX_Y
        if target_pos[2] < MIN_Z: target_pos[2] = MIN_Z
            
        xy_dist = np.linalg.norm(target_pos[:2])
        if xy_dist < MIN_RADIUS_XY:
            if xy_dist < 1e-6: 
                target_pos[0] = 0
                target_pos[1] = -MIN_RADIUS_XY
            else:
                scale = MIN_RADIUS_XY / xy_dist
                target_pos[0] *= scale
                target_pos[1] *= scale
        
        dist = np.linalg.norm(target_pos)
        if dist > MAX_RADIUS:
            target_pos *= (MAX_RADIUS / dist)

        if not np.array_equal(old_pos, target_pos):
             msg = "🔒 Clamped"
             if old_pos[1] > MAX_Y: msg = "🔒 Wall (Y+)"
             elif np.linalg.norm(old_pos[:2]) < MIN_RADIUS_XY: msg = "🔒 Inner Cyl"
             elif np.linalg.norm(old_pos) > MAX_RADIUS: msg = "🔒 Outer Sph"
             print(f"{msg}", end='\r')

        q, debug_msg, cond = sim.solve_pos_ik(target_pos, q)
        
        sim.viz.display(q)
        sim.viz.viewer["target"].set_transform(pin.SE3(np.eye(3), target_pos).homogeneous)
        
        log_str = (f"{debug_msg} | Tgt:[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] | "
                   f"R_xy:{np.linalg.norm(target_pos[:2]):.2f}m")

        print(log_str, end='\r')
        
        log_counter += 1
        if log_counter % 10 == 0 or cond > 50 or "Diverged" in debug_msg:
             logger.info(log_str)

        clock.tick(FREQ)

    pygame.quit()
    logger.info("🛑 仿真结束")

if __name__ == "__main__":
    main()