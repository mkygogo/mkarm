import time
import torch
import numpy as np
import logging
import sys
import os
import argparse
import json
import cv2
from pathlib import Path

# 路径修正
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))
sys.path.append("./src")

# [更新] 导入必要的 Config 类
from lerobot.robots.mkrobot.mk_robot import MKRobot, MKRobotConfig
from lerobot.teleoperators.gamepad.gamepad_ik_teleop import GamepadIKTeleop, GamepadIKTeleopConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig # 显式导入相机配置

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # 如果想要更详细的信息可以改为 "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("Pi05-Recorder")

# 按键定义 (Xbox)
# BTN_A = 0  # Start Recording
BTN_Y = 3  # Success & Finish (Hold to mark success, Release to save & home)
# BTN_X = 2  # Fail & Reset (Hold to home)


# 为了归一化，根据mk_robot.py中的限位值算的
"""
我们可以定义如下的 Scale_Factor（取物理限位区间的最大绝对值）：
J1: $max(|-3.0|, |3.0|) = 3.0$
J2: $max(|0.0|, |3.0|) = 3.0$
J3: $max(|0.0|, |3.0|) = 3.0$
J4: $max(|-1.7|, |1.2|) = 1.7$
J5: $max(|-0.4|, |0.4|) = 0.4$
J6: $max(|-2.0|, |2.0|) = 2.0$
Gripper: $1.0$ (假设夹爪已经是 $0-1$ 范围)
"""
# 这里的数值取自你提供的安全配置中每个关节的最大绝对值
JOINT_NORM_SCALE = np.array([3.0, 3.0, 3.0, 1.7, 0.4, 2.0, 1.0])

def process_to_square(img, target_size=448):
    """ 将图像裁剪并缩放为正方形。兼容 [C, H, W] 张量或 Numpy 数组 """
    if isinstance(img, torch.Tensor):
        # 转换 [C, H, W] -> [H, W, C] 供 OpenCV 处理
        img_np = img.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = img

    h, w = img_np.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_cropped = img_np[start_h:start_h + min_dim, start_w:start_w + min_dim]
    
    img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # 转换为 uint8 并转回 [C, H, W] 张量
    if img_resized.max() <= 1.0:
        img_resized = (img_resized * 255).astype(np.uint8)
    return torch.from_numpy(img_resized.astype(np.uint8)).permute(2, 0, 1)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="env_config_gamepad_record_data.json")
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = json.load(f)
    repo_id = cfg['dataset']['repo_id']
    root_dir = Path(cfg['dataset']['root'])
    fps = cfg['env']['fps']
    task_label = cfg['dataset']['task']
    
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": ["j1","j2","j3","j4","j5","j6","gripper"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["j1","j2","j3","j4","j5","j6","gripper"]},
        "observation.images.top": {"dtype": "video", "shape": [3, 448, 448], "names": ["c", "h", "w"]},
        "observation.images.wrist": {"dtype": "video", "shape": [3, 448, 448], "names": ["c", "h", "w"]},
    }

    print(f"🚀 准备采集数据: {repo_id}")

    # --- 1. 数据集初始化 ---
    if (root_dir / "meta/info.json").exists():
        print(f"🔄 检测到现有数据集，正在加载...")
        dataset = LeRobotDataset(repo_id=repo_id, root=root_dir)
        print(f"✅ 加载成功！接续从 Episode {dataset.num_episodes} 开始。")
    else:
        print(f"🆕 未检测到数据集，正在创建新数据集...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root_dir,
            fps=fps,
            robot_type=cfg['env']['robot']['type'],
            features=features
        )
    episode_idx = dataset.num_episodes

    # --- 2. 初始化 Robot (参考 run_real_teleop.py) ---
    robot_json_cfg = cfg['env']['robot']
    
    # [更新] 使用 Config 对象初始化
    mk_robot_config = MKRobotConfig(
        port=robot_json_cfg['port'],
        joint_velocity_scaling=1.0  # 默认设置为 1.0，如需限制速度可调低
    )
    robot = MKRobot(mk_robot_config)

    # 初始化相机 (保持原有逻辑，因为 run_real_teleop.py 通常不带相机)
    for name, cam_cfg in robot_json_cfg['cameras'].items():
        cam_config = OpenCVCameraConfig(
            index_or_path=cam_cfg['index_or_path'], 
            fps=cam_cfg['fps'], 
            width=cam_cfg['width'], 
            height=cam_cfg['height']
        )
        robot.cameras[name] = OpenCVCamera(cam_config)
    
    robot.connect()
    print(f"✅ 真机与相机连接成功")

    # --- 3. 初始化 Teleop (核心修改) ---
    teleop_json_cfg = cfg['env']['teleop']

    # [更新] 先创建 Config 对象，显式包含速度参数
    teleop_config = GamepadIKTeleopConfig(
        type="gamepad_ik",
        urdf_path=teleop_json_cfg['urdf_path'],
        mesh_dir=teleop_json_cfg['mesh_dir'],
        fps=fps,
        visualize=teleop_json_cfg.get('visualize', True),
        inverse_kinematics=teleop_json_cfg.get('inverse_kinematics', {}),
        # 可以在这里调整速度，覆盖默认值
        trans_speed=teleop_json_cfg.get('trans_speed', 0.002), 
        rot_speed=teleop_json_cfg.get('rot_speed', 0.02)
    )

    # [更新] 传入 config 参数，解决 'AttributeError: NoneType has no attribute id'
    teleop = GamepadIKTeleop(
        config=teleop_config,  # <--- 关键修改：传入 config 对象
        urdf_path=teleop_config.urdf_path,
        mesh_dir=teleop_config.mesh_dir,
        fps=teleop_config.fps,
        visualize=teleop_config.visualize,
        inverse_kinematics=teleop_config.inverse_kinematics
    )
    teleop.connect()
    print("✅ 手柄与 IK 核心就绪")

    print("\n" + "="*50)
    print("🎮 操作说明:")
    print("   [RB 按住] : 激活控制 (IK)")
    print("   [A 键]    : ▶️ 开始录制 (Start)")
    print("   [Y 键]    : ✅ 保存并归位")
    print("   [B 键]    : ✅ 结束退出")
    print("   [x 键]    : ♻️ 长按归位 (视为失败并保存/丢弃)")
    print("="*50 + "\n")
  
    episode_data = None

    
    MAX_TIME_S = cfg['processor']['reset']['control_time_s']
    episode_start_time = 0

    try:
        while episode_idx < cfg['dataset']['num_episodes_to_record']:
            # --- 阶段 1: 等待按下 A 键开始录制 ---
            logger.info(f"⌛ 等待开始录制 Episode {episode_idx} (按 A 键)...")
            while True:
                obs = robot.get_observation()
                # 兼容 Teleop 返回字典的情况
                action_out = teleop.get_action(obs)
                action_tensor = action_out["action"] if isinstance(action_out, dict) else action_out
                robot.send_action(action_tensor)
                
                events = teleop.get_teleop_events()
                if events.get(TeleopEvents.START_RECORDING): # A 键
                    break
                if events.get(TeleopEvents.TERMINATE_EPISODE): # B 键
                    return # 直接退出 main
                time.sleep(1.0 / fps)

            # --- 阶段 2: 录制中 ---
            logger.info(f"🔴 录制中 Episode {episode_idx} (按 Y 键保存)...")
            episode_data = {k: [] for k in ["observation.images.top", "observation.images.wrist", "observation.state", "action"]}
            
            while True:
                loop_start = time.time()
                obs = robot.get_observation()
                action_out = teleop.get_action(obs)
                action_tensor = action_out["action"] if isinstance(action_out, dict) else action_out
                robot.send_action(action_tensor)
                
                # 【修复 KeyError】使用 obs.keys() 提供的完整路径
                episode_data["observation.state"].append(obs["observation.state"].cpu().numpy())
                episode_data["action"].append(action_tensor.cpu().numpy())
                episode_data["observation.images.top"].append(obs["observation.images.top"])
                episode_data["observation.images.wrist"].append(obs["observation.images.wrist"])
                
                events = teleop.get_teleop_events()
                
                # --- 阶段 3: 保存逻辑 (Y 键) ---
                if events.get(TeleopEvents.SUCCESS):
                    logger.info(f"💾 正在处理并保存 Episode {episode_idx}...")
                    num_frames = len(episode_data["action"])
                    for i in range(num_frames):
                        norm_state = episode_data["observation.state"][i] / JOINT_NORM_SCALE
                        norm_action = episode_data["action"][i] / JOINT_NORM_SCALE
                        norm_state = np.clip(norm_state, -1.0, 1.0)
                        norm_action = np.clip(norm_action, -1.0, 1.0)
                        frame = {
                            "observation.state": norm_state.astype(np.float32),
                            "action": norm_action.astype(np.float32),
                            "observation.images.top": process_to_square(episode_data["observation.images.top"][i]),
                            "observation.images.wrist": process_to_square(episode_data["observation.images.wrist"][i]),
                            "task": cfg['dataset']['task']
                        }
                        dataset.add_frame(frame)
                    
                    dataset.save_episode()
                    logger.info(f"✅ Episode {episode_idx} 已保存！")
                    episode_idx = dataset.num_episodes
                    
                    # 串口排空 & 自动归位 (您要求的稳定归零逻辑)
                    print("🧹 清理串口...", end="")
                    flush_start = time.time()
                    while time.time() - flush_start < 1.0:
                        try:
                            _ = robot.get_observation()
                            _ = teleop.get_action(_)
                        except: pass
                        time.sleep(1.0/fps)
                    print("完成")
                    
                    if not teleop.core.is_homing:
                        teleop.core.start_homing()
                    break # 跳出录制循环，回到等待 A 键阶段

                if events.get(TeleopEvents.TERMINATE_EPISODE): # B 键
                    return

                # 频率控制
                dt = time.time() - loop_start
                time.sleep(max(0, (1.0/fps) - dt))

    finally:
        dataset.finalize()
        robot.disconnect()
        teleop.disconnect()    



if __name__ == "__main__":
    main()