import time
import torch
import numpy as np
import logging
import sys
import argparse

# 确保能导入模块
sys.path.append("./src")
from lerobot.teleoperators.utils import TeleopEvents
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
        config=teleop_config,
        inverse_kinematics={}
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

            # 1. 获取观测
            observation = robot.get_observation()
            
            # 2. [新增] 获取手柄事件并处理业务逻辑
            events = teleop.get_teleop_events()
            
            # 处理归位请求 (X键长按)
            if events[TeleopEvents.RERECORD_EPISODE]:
                # 防止重复触发：只有当前不在归位时才触发
                if not teleop.core.is_homing:
                    print("🔄 检测到重置信号 (X)，开始归位...")
                    teleop.core.start_homing()

            # 3. 计算动作 (get_action 内部会处理: 如果 is_homing=True 则返回归位轨迹，否则返回 IK/吸附)
            action = teleop.get_action(observation)

            # 4. 发送动作
            robot.send_action(action)

            # ... (保持频率控制代码不变) ...
            dt = time.time() - start_time
            if dt < 1.0 / 60:
                time.sleep(1.0 / 60 - dt)

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