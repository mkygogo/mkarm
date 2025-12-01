import time
import torch
from lerobot.teleoperators.gamepad.gamepad_ik_teleop import GamepadIKTeleop

def main():
    # ⚠️ 修改这里为你实际的路径
    URDF_PATH = "../hardware/urdf/urdf/dk2.SLDASM.urdf"
    MESH_DIR = "../hardware/urdf/meshes"

    print("🚀 初始化 Gamepad IK Teleop (Sim Mode)...")
    
    # 实例化 Teleop，开启 visualize=True
    teleop = GamepadIKTeleop(
        urdf_path=URDF_PATH,
        mesh_dir=MESH_DIR,
        fps=60,
        visualize=True
    )

    teleop.connect()
    
    print("✅ 就绪！请打开浏览器查看 Meshcat 可视化。")
    print("🎮 按下手柄控制：左摇杆移动 XY，右摇杆移动 Z，按键 A/B 控制夹爪。")
    print("按 Ctrl+C 退出。")

    try:
        while True:
            # 模拟 LeRobot 的循环
            start_time = time.time()
            
            # 1. 获取动作 (内部会自动跑 IK 并更新 Meshcat)
            # 这里的 observation 传个空字典就行，因为我们是纯 IK 遥操，不依赖环境反馈
            action = teleop.get_action(observation={})
            
            # 打印一下动作看看 (关节角度)
            # print(f"Action: {action.numpy().round(2)}")
            
            # 2. 维持 60Hz 循环
            dt = time.time() - start_time
            sleep_time = max(0, (1.0 / 60.0) - dt)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 停止运行...")
    finally:
        teleop.disconnect()

if __name__ == "__main__":
    main()