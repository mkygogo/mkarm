import time
import torch
import numpy as np
import logging
import sys
import argparse

# 确保能导入模块
sys.path.append("./src")

from lerobot.robots.mkrobot.mk_robot import MKRobot, MKRobotConfig
from lerobot.teleoperators.gamepad.gamepad_ik_teleop import GamepadIKTeleop, GamepadIKTeleopConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RealTeleop")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="真机串口")
    args = parser.parse_args()
    URDF_PATH = "../hardware/urdf/urdf/dk2.SLDASM.urdf"
    MESH_DIR = "../hardware/urdf/meshes"

    print("🚀 初始化真机 Teleop 系统...")

    # 1. 初始化 Robot (硬件层)
    # MKRobot 会负责处理 HARDWARE_DIR 和电机通信
    robot_config = MKRobotConfig(
        port=args.port,
        joint_velocity_scaling=1.0
    )
    try:
        robot = MKRobot(robot_config)
        robot.connect()
        print(f"✅ 真机连接成功: {args.port}")
    except Exception as e:
        print(f"❌ 真机连接失败: {e}")
        return

    # 2. 初始化 Teleop (算法层)
    # GamepadIKTeleop 会启动 Pygame 和 Meshcat
    teleop_config = GamepadIKTeleopConfig(
        urdf_path=URDF_PATH,
        mesh_dir=MESH_DIR, # 假设 mesh 在这里
        fps=60,
        visualize=True
    )
    teleop = GamepadIKTeleop(
        urdf_path=URDF_PATH,
        mesh_dir=MESH_DIR,
        fps=teleop_config.fps,
        visualize=teleop_config.visualize,
        config=teleop_config
    )
    teleop.connect()
    print("✅ 手柄与 IK 核心就绪")

    print("\n⚠️  警告: 机械臂即将开始同步！")
    print("👉 请确保急停按钮在手边。")
    print("👉 按 Ctrl+C 退出程序。\n")
    
    #input("按 [Enter] 键开始控制循环...")

    try:
        while True:
            start_time = time.time()

            # --- A. 获取真机状态 ---
            # robot.get_observation() 会返回 Sim 坐标系下的关节角度
            observation = robot.get_observation()
            
            # --- B. 计算 IK 动作 ---
            # Teleop 内部逻辑：
            # - 如果手柄没动 -> set_state_from_hardware (吸附真机位置)
            # - 如果手柄动了 -> step (从当前位置开始 IK)
            action = teleop.get_action(observation)

            # --- C. 发送动作给真机 ---
            # action 是 Sim 坐标系动作，robot.send_action 会自动转为电机指令
            robot.send_action(action)

            # --- D. 维持频率 ---
            dt = time.time() - start_time
            sleep_time = max(0, (1.0 / 60.0) - dt)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 用户停止...")
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
    finally:
        print("正在断开连接...")
        robot.disconnect()
        teleop.disconnect()
        print("已安全退出。")

if __name__ == "__main__":
    main()