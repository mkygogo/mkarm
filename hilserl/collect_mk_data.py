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

from lerobot.robots.mkrobot.mk_robot import MKRobot, MKRobotConfig
from lerobot.teleoperators.gamepad.gamepad_ik_teleop import GamepadIKTeleop, GamepadIKTeleopConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataCollector")

# 按键定义 (Xbox)
BTN_A = 0  # Start Recording
BTN_Y = 3  # Success & Finish (Hold to mark success, Release to save & home)
BTN_X = 2  # Fail & Reset (Hold to home)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="env_config_gamepad_record_data.json")
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = json.load(f)
    repo_id = cfg['dataset']['repo_id']
    root_dir = Path(cfg['dataset']['root'])
    fps = cfg['env']['fps']
    wrist_h = cfg['env']['robot']['cameras']['wrist_camera']['height']
    wrist_w = cfg['env']['robot']['cameras']['wrist_camera']['width']
    side_h = cfg['env']['robot']['cameras']['side_camera']['height']
    side_w = cfg['env']['robot']['cameras']['side_camera']['width']

    print(f"🚀 准备采集数据: {repo_id}")

    # [修改] 检查是否已有数据集，有则加载，无则创建
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
    #更新 episode_idx 计数器
    episode_idx = dataset.num_episodes

    # Init Robot
    robot_cfg = cfg['env']['robot']
    robot = MKRobot(MKRobotConfig(port=robot_cfg['port']))
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    for name, cam_cfg in robot_cfg['cameras'].items():
        robot.cameras[name] = OpenCVCamera(OpenCVCameraConfig(
            index_or_path=cam_cfg['index_or_path'], fps=cam_cfg['fps'], 
            width=cam_cfg['width'], height=cam_cfg['height']))
    robot.connect()

    # Init Teleop
    teleop_cfg = cfg['env']['teleop']
    teleop = GamepadIKTeleop(
        urdf_path=teleop_cfg['urdf_path'],
        mesh_dir=teleop_cfg['mesh_dir'],
        fps=fps,
        visualize=teleop_cfg.get('visualize', True),
        inverse_kinematics=teleop_cfg.get('inverse_kinematics', {})
    )
    teleop.connect()

    print("\n" + "="*50)
    print("🎮 操作说明:")
    print("   [RB 按住] : 激活控制 (IK)")
    print("   [A 键]    : ▶️ 开始录制 (Start)")
    print("   [Y 键]    : ✅ 按住=成功; 松开=保存并归位")
    print("   [X 键]    : ♻️ 长按归位 (视为失败并保存)")
    print("="*50 + "\n")

    is_recording = False
    is_success = False # 当前 Episode 是否成功
    y_was_pressed = False # Y键状态追踪
    
    episode_data = {k: [] for k in ["observation.images.wrist_camera", "observation.images.side_camera", "observation.state", "observation.velocity", "action", "success"]}
    prev_state = None
    
    # 超时设置
    MAX_TIME_S = cfg['processor']['reset']['control_time_s']
    episode_start_time = 0

    try:
        while True:
            loop_start = time.time()
            obs = robot.get_observation()
            images = robot.capture_images()
            curr_state = obs['observation.state']
            if prev_state is None: prev_state = curr_state
            velocity = (curr_state - prev_state) * fps
            prev_state = curr_state

            action = teleop.get_action(obs)
            robot.send_action(action)

            # --- 录制状态机 ---
            
            # 1. 开始录制 (A)
            if teleop.joystick.get_button(BTN_A) and not is_recording:
                print(f"\n🔴 [Ep {episode_idx}] 开始录制...")
                is_recording = True
                is_success = False
                y_was_pressed = False
                episode_start_time = time.time()
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

            # 2. 录制中逻辑
            if is_recording:
                # 获取当前这一帧 Y 键是否按下
                current_frame_success = teleop.joystick.get_button(BTN_Y)
                # 记录数据
                episode_data["observation.state"].append(curr_state.cpu().numpy())
                episode_data["observation.velocity"].append(velocity.cpu().numpy())
                episode_data["action"].append(action.cpu().numpy())
                episode_data["observation.images.wrist_camera"].append(images["wrist_camera"])
                episode_data["observation.images.side_camera"].append(images["side_camera"])  
                episode_data["success"].append(current_frame_success)#记录这一帧是否成功到列表中

                # Y键逻辑：按住即标记成功
                if current_frame_success:
                    is_success = True
                    y_was_pressed = True
                    sys.stdout.write(f"\r✅ [SUCCESS] Rec: {len(episode_data['action'])} frames")
                else:
                    sys.stdout.write(f"\r🔴 [recording] Rec: {len(episode_data['action'])} frames")
                sys.stdout.flush()

                # 结束条件 1: 松开 Y 键 (下降沿)
                if y_was_pressed and not teleop.joystick.get_button(BTN_Y):
                    print(f"\n💾 Y键释放 -> 保存 (Success={is_success}) 并归位...")
                    save_and_reset = True
                
                # 结束条件 2: 超时
                elif (time.time() - episode_start_time) > MAX_TIME_S:
                    print(f"\n⏰ 超时 ({MAX_TIME_S}s) -> 保存 (Success={is_success}) 并归位...")
                    save_and_reset = True
                
                # 结束条件 3: 归位中断 (X键长按或 Teleop 内部触发了 Homing)
                elif teleop.core.is_homing: 
                    print(f"\n♻️ 检测到归位 -> 中断保存 (Success={is_success})...")
                    save_and_reset = True
                
                else:
                    save_and_reset = False

                # 执行保存与复位
                if save_and_reset:
                    is_recording = False
                    
                    # [HIL-SERL 核心修改] 写入符合 RL 标准的数据
                    # 我们遍历这一集的所有帧，逐帧打标签
                    num_frames = len(episode_data['action'])

                    # 写入 Dataset
                    for i in range(num_frames):
                        # 判断是否是这一集的最后一帧
                        is_last_frame = (i == num_frames - 1)
                        
                        # 1. Done: 只有最后一帧是 True
                        done = is_last_frame
                        
                        # 2. Success: 从我们刚才记录的列表里取值
                        # 只要录制那一刻你按着 Y，这一帧就是 True
                        frame_success = episode_data["success"][i]
                        
                        # 3. Reward: 对应 Success，按着就是 1.0，没按就是 0.0
                        reward = 1.0 if frame_success else 0.0

                        frame = {
                            "observation.images.wrist_camera": episode_data["observation.images.wrist_camera"][i],
                            "observation.images.side_camera": episode_data["observation.images.side_camera"][i],
                            "observation.state": episode_data["observation.state"][i],
                            "observation.velocity": episode_data["observation.velocity"][i],
                            "action": episode_data["action"][i],
                            "task": cfg['dataset']['task'],
                            #必须包含这三个字段才能跑 HIL-SERL
                            "next.reward": np.array([reward], dtype=np.float32),
                            "next.done": np.array([done], dtype=bool),
                            "next.success": np.array([frame_success], dtype=bool)
                        }
                        dataset.add_frame(frame)
                    
                    # 保存 Episode (带上成功标记，LeRobot 是否支持取决于 meta)
                    # 我们把 success 状态打印出来，HIL-SERL 可能需要后续处理这个标记
                    # 目前 LeRobotDataset 还没有标准的 is_success 字段，通常通过 task 名字区分?
                    # 或者我们可以 hack 一下，把 success 状态写在 episode 的 info 里?
                    # 暂时先正常保存。
                    dataset.save_episode()
                    print(f"✅ Episode {episode_idx} Saved.")
                    episode_idx = dataset.num_episodes
                    
                    #排空串口缓冲区 (Flush Serial Buffer)
                    # 关键修复：在等待期间必须保持 teleop 活跃！
                    print("🧹 正在排空过期的串口数据...", end="")
                    flush_start = time.time()
                    while time.time() - flush_start < 1.0: # 读 1 秒
                        try:
                            # 1. 读取机器人最新状态 (使用标准API)
                            obs = robot.get_observation()
                            # 2. [关键] 持续更新手柄状态
                            # 这能处理排队的手柄事件(如松开按键)，并防止 dt 计算错误
                            teleop.get_action(obs)
                        except:
                            pass
                        # 维持正常的循环频率，防止死循环占用 CPU
                        time.sleep(1.0 / fps)
                    print(" 完成！")
                    
                    # 再次获取一次最新的 Observation 来校准 prev_state
                    # 防止速度计算出现巨大的跳变
                    obs = robot.get_observation()
                    prev_state = obs['observation.state']

                    # 触发自动归位 (如果不是因为已经在归位而触发的)
                    if not teleop.core.is_homing:
                        teleop.core.start_homing()

            dt = time.time() - loop_start
            time.sleep(max(0, (1.0/fps) - dt))

    except KeyboardInterrupt:
        print("\n🛑 停止采集。")
    finally:
        robot.disconnect()
        teleop.disconnect()

if __name__ == "__main__":
    main()