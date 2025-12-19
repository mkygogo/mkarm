import time
import torch
import numpy as np
import logging
import sys
import os
import argparse
import json
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataCollector")

# 按键定义 (Xbox)
# BTN_A = 0  # Start Recording
BTN_Y = 3  # Success & Finish (Hold to mark success, Release to save & home)
# BTN_X = 2  # Fail & Reset (Hold to home)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="env_config_gamepad_record_data.json")
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = json.load(f)
    repo_id = cfg['dataset']['repo_id']
    root_dir = Path(cfg['dataset']['root'])
    fps = cfg['env']['fps']
    
    # 获取相机分辨率参数
    wrist_h = cfg['env']['robot']['cameras']['wrist_camera']['height']
    wrist_w = cfg['env']['robot']['cameras']['wrist_camera']['width']
    side_h = cfg['env']['robot']['cameras']['side_camera']['height']
    side_w = cfg['env']['robot']['cameras']['side_camera']['width']

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
            features={
                "observation.images.wrist_camera": 
                            {"dtype": "video", "shape": (wrist_h, wrist_w, 3), "names": ["height", "width", "channel"]},
                "observation.images.side_camera": 
                            {"dtype": "video", "shape": (side_h, side_w, 3), "names": ["height", "width", "channel"]},
                "observation.state": 
                            {"dtype": "float32", "shape": (7,), "names": ["j1","j2","j3","j4","j5","j6","grp"]},
                "observation.velocity": 
                            {"dtype": "float32", "shape": (7,), "names": ["j1","j2","j3","j4","j5","j6","grp"]},
                "action": 
                            {"dtype": "float32", "shape": (7,), "names": ["j1","j2","j3","j4","j5","j6","grp"]},
                "next.reward": 
                            {"dtype": "float32", "shape": (1,), "names": None},
                "next.done": 
                            {"dtype": "bool", "shape": (1,), "names": None},
                "next.success": 
                            {"dtype": "bool", "shape": (1,), "names": None},
            }
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
    print("   [Y 键]    : ✅ 按住=成功; 松开=保存并归位")
    print("   [X 键]    : ♻️ 长按归位 (视为失败并保存/丢弃)")
    print("="*50 + "\n")

    is_recording = False
    is_success = False 
    y_was_pressed = False
    
    episode_data = None
    prev_state = None
    
    MAX_TIME_S = cfg['processor']['reset']['control_time_s']
    episode_start_time = 0

    try:
        print(f"\n⏳ 等待开始 Episode {episode_idx}... (按 A 开始)")
        while True:
            loop_start = time.time()
            
            # 1. 获取观测
            obs = robot.get_observation()
            
            # 2. 获取手柄事件 (用于状态机)
            events = teleop.get_teleop_events()

            # 3. 图像捕获
            images = robot.capture_images()
            curr_state = obs['observation.state']
           
            if prev_state is None: 
                prev_state = curr_state
            
            velocity = (curr_state - prev_state) * fps
            prev_state = curr_state

            # 4. 计算并发送动作
            action = teleop.get_action(obs)
            robot.send_action(action)

            # ================= [修改开始] =================
            # 【新增】全局回零检测 (无论是否在录制，都允许回零)
            # 把它放在 state machine 之前
            if events[TeleopEvents.RERECORD_EPISODE]:
                # 防止重复触发
                if not teleop.core.is_homing:
                    print(f"\n🔄 检测到重置信号 (X) -> 正在归位...")
                    teleop.core.start_homing()
                    
                    # 如果正在录制，需要强制中断录制
                    if is_recording:
                        print("   (中断当前录制，数据丢弃)")
                        is_recording = False
                        save_and_reset = False # 确保不进入保存流程
            # ================= [修改结束] =================

            # --- 录制状态机 ---
            
            # [Trigger] 开始录制 (A键)
            if events[TeleopEvents.SUCCESS] and not is_recording and not teleop.core.is_homing:
                print(f"\n🔴 [Ep {episode_idx}] 开始录制...")
                is_recording = True
                is_success = False
                y_was_pressed = False
                episode_start_time = time.time()
                # 重置缓存列表
                episode_data = {
                    k: [] for k in [
                        "observation.images.wrist_camera", 
                        "observation.images.side_camera", 
                        "observation.state", 
                        "observation.velocity", 
                        "action",
                        "success"  
                    ]
                }

            # [Process] 录制中逻辑
            if is_recording:
                current_frame_success = teleop.joystick.get_button(BTN_Y)
                
                # 数据追加
                episode_data["observation.state"].append(curr_state.cpu().numpy())
                episode_data["observation.velocity"].append(velocity.cpu().numpy())
                episode_data["action"].append(action.cpu().numpy())
                episode_data["observation.images.wrist_camera"].append(images["wrist_camera"])
                episode_data["observation.images.side_camera"].append(images["side_camera"])  
                episode_data["success"].append(current_frame_success)

                # 实时反馈录制状态
                if current_frame_success:
                    is_success = True
                    y_was_pressed = True
                    sys.stdout.write(f"\r✅ [SUCCESS] Rec: {len(episode_data['action'])} frames")
                else:
                    sys.stdout.write(f"\r🔴 [recording] Rec: {len(episode_data['action'])} frames")
                sys.stdout.flush()

                save_and_reset = False

                # [Interrupt 1] 重置信号 (X键长按) -> 丢弃数据并归位
                if events[TeleopEvents.RERECORD_EPISODE]:
                    print(f"\n❌ 检测到重置信号 (X) -> 丢弃数据并归位")
                    is_recording = False
                    teleop.core.start_homing()
                    save_and_reset = False 

                # [End 1] 松开 Y 键 (任务完成)
                elif y_was_pressed and not teleop.joystick.get_button(BTN_Y):
                    print(f"\n💾 Y键释放 -> 保存 (Success={is_success}) 并归位...")
                    save_and_reset = True
                    teleop.core.start_homing()
                
                # [End 2] 超时
                elif (time.time() - episode_start_time) > MAX_TIME_S:
                    print(f"\n⏰ 超时 ({MAX_TIME_S}s) -> 保存 (Success={is_success}) 并归位...")
                    save_and_reset = True
                    teleop.core.start_homing()
                
                # [End 3] 意外归位 (安全机制触发)
                elif teleop.core.is_homing: 
                    print(f"\n♻️ 检测到归位 -> 中断保存 (Success={is_success})...")
                    save_and_reset = True
                
                # --- 保存逻辑 ---
                if save_and_reset:
                    is_recording = False
                    num_frames = len(episode_data['action'])

                    # 写入 Dataset (逐帧写入，适配 HIL-SERL 格式)
                    for i in range(num_frames):
                        is_last_frame = (i == num_frames - 1)
                        done = is_last_frame
                        frame_success_bool = episode_data["success"][i]
                        reward = 1.0 if frame_success_bool else 0.0

                        frame = {
                            "observation.images.wrist_camera": episode_data["observation.images.wrist_camera"][i],
                            "observation.images.side_camera": episode_data["observation.images.side_camera"][i],
                            "observation.state": episode_data["observation.state"][i],
                            "observation.velocity": episode_data["observation.velocity"][i],
                            "action": episode_data["action"][i],
                            "task": cfg['dataset']['task'],
                            "next.reward": np.array([reward], dtype=np.float32),
                            "next.done": np.array([done], dtype=bool),
                            "next.success": np.array([frame_success_bool], dtype=bool)
                        }
                        dataset.add_frame(frame)
                    
                    dataset.save_episode()
                    print(f"✅ Episode {episode_idx} Saved.")
                    episode_idx = dataset.num_episodes
                    
                    # 排空串口缓冲区，防止下一集开始时读到旧数据
                    print("🧹 正在排空过期的串口数据...", end="")
                    flush_start = time.time()
                    while time.time() - flush_start < 1.0: 
                        try:
                            # 必须持续调用 get_action 保持 Teleop 和 Robot 通信活跃
                            obs_tmp = robot.get_observation()
                            teleop.get_action(obs_tmp)
                        except:
                            pass
                        time.sleep(1.0 / fps)
                    print(" 完成！")
                    
                    # 重新校准 prev_state
                    obs = robot.get_observation()
                    prev_state = obs['observation.state']

                    # 确保归位
                    if not teleop.core.is_homing:
                        teleop.core.start_homing()

            # 频率控制
            dt = time.time() - loop_start
            time.sleep(max(0, (1.0/fps) - dt))

    except KeyboardInterrupt:
        print("\n🛑 停止采集。")
    finally:
        robot.disconnect()
        teleop.disconnect()

if __name__ == "__main__":
    main()