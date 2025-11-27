import time
import numpy as np
import pygame
from follower_mkarm import MKFollower, MKFollowerConfig

NOTICE_STR = """
***********************************
这个就是为了测试机械臂每个轴都能正常工作
映射关系：
joint1 左摇杆左右控制
joint2 左摇杆前后控制
joint3 右摇杆前后控制
joint4 十字键前后控制
joint5 右摇杆左右控制
joint6 十字键左右控制
gripper RT/LT
***********************************
"""

# --- 配置 ---
FOLLOWER_PORT = "/dev/ttyACM0"
FREQ = 60  # 控制频率 Hz

# 关节速度 (弧度/tick) - 对应你仿真里的 SPEED
# 真机如果觉得太快，可以把这个数值调小
SPEED_J1_J3 = 0.01
SPEED_J4_J6 = 0.01
GRIPPER_SPEED = 0.02 # 夹爪开合速度

# 关节软限位 (基于 dk2.SLDASM.urdf 但是根据实际情况做了调整)
# [Min, Max]
JOINT_LIMITS = {
    "joint_1": [-3.0, 3.0],
    "joint_2": [-0.3, 3.0],
    "joint_3": [0.0, 3.0],
    "joint_4": [-1.7, 1.2],
    "joint_5": [-0.4, 0.4],
    "joint_6": [-2.0, 2.0]
}

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# --- 直接复用你的仿真辅助函数 ---

def filter_stick(val_1, val_2, deadzone=0.15, snap_ratio=0.4):
    """ 处理双轴摇杆的死区和防抖 """
    if abs(val_1) < deadzone: val_1 = 0
    if abs(val_2) < deadzone: val_2 = 0
    
    if val_1 != 0 and val_2 != 0:
        abs_1 = abs(val_1)
        abs_2 = abs(val_2)
        if abs_2 < abs_1 * snap_ratio:
            val_2 = 0
        elif abs_1 < abs_2 * snap_ratio:
            val_1 = 0
    return val_1, val_2

def get_gamepad_deltas(joystick):
    pygame.event.pump()
    
    d_q = np.zeros(6) 
    d_gripper = 0
    
    # 读取原始数据
    raw_lx = joystick.get_axis(0)
    raw_ly = joystick.get_axis(1)
    raw_rx = joystick.get_axis(3)
    raw_ry = joystick.get_axis(4) 

    # 摇杆死区处理
    lx, ly = filter_stick(raw_lx, raw_ly, deadzone=0.15, snap_ratio=0.5)
    rx, ry = filter_stick(raw_rx, raw_ry, deadzone=0.15, snap_ratio=0.5)
    
    # --- 映射逻辑 (基于你提供的最新代码) ---
    
    # 1. 左摇杆左右 -> Joint 1
    d_q[0] = lx * SPEED_J1_J3
    
    # 2. 左摇杆前后 -> Joint 2
    d_q[1] = ly * SPEED_J1_J3

    # 3. 右摇杆前后 -> Joint 3
    d_q[2] = ry * SPEED_J1_J3

    # 4. 十字键上下 -> Joint 4
    hat_y = -joystick.get_hat(0)[1]
    if hat_y != 0:
        d_q[3] = hat_y * SPEED_J4_J6
    
    # 5. 右摇杆左右 -> Joint 5 (注意：你之前的代码里此处用了 rx)
    d_q[4] = -rx * SPEED_J4_J6

    # 6. 十字键左右 -> Joint 6 (注意：你之前的代码里此处用了 hat_x)
    hat_x = joystick.get_hat(0)[0]
    if hat_x != 0:
        d_q[5] = -hat_x * SPEED_J4_J6

    # 7. 夹爪 (RT/LT)
    rt_val = (joystick.get_axis(5) + 1) / 2
    lt_val = (joystick.get_axis(2) + 1) / 2
    
    if rt_val > 0.1:
        d_gripper = 1   # Close
    elif lt_val > 0.1:
        d_gripper = -1  # Open
    
    return -d_q, d_gripper

# --- 主程序 ---

def main():
    # 1. 初始化 Pygame 手柄
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("❌ 未检测到手柄！")
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"🎮 已连接手柄: {joystick.get_name()}")

    # 2. 连接机械臂
    print("🤖 连接机械臂中...")
    try:
        config = MKFollowerConfig(
            port=FOLLOWER_PORT, 
            joint_velocity_scaling=1.0 
        )
        bot = MKFollower(config)
        bot.connect()
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 3. 初始化目标状态
    print("📡 读取初始状态...")
    try:
        obs = bot.get_observation()
    except Exception as e:
        print(f"❌ 读取状态失败: {e}")
        bot.disconnect()
        return

    # 维护一个目标关节角度字典 (积分控制用)
    target_joints = {
        "joint_1": obs["joint_1.pos"],
        "joint_2": obs["joint_2.pos"],
        "joint_3": obs["joint_3.pos"],
        "joint_4": obs["joint_4.pos"],
        "joint_5": obs["joint_5.pos"],
        "joint_6": obs["joint_6.pos"],
    }
    
    # 夹爪状态: 0.0 (Open) ~ 1.0 (Close)
    # 尝试从观测中读取当前夹爪状态，如果没有则默认 0
    current_gripper = obs.get("gripper.pos", 0.0)
    print(NOTICE_STR)
    print("✅ 开始控制！按 Ctrl+C 退出")
    print("-" * 60)
    print(f"{'J1':^8}|{'J2':^8}|{'J3':^8}|{'J4':^8}|{'J5':^8}|{'J6':^8}|{'Grip':^6}")
    print("-" * 60)

    clock = pygame.time.Clock()
    
    try:
        while True:
            # 1. 获取增量
            d_q, d_gripper = get_gamepad_deltas(joystick)
            
            # 2. 更新关节 (积分 + 限位)
            # d_q 的顺序对应 J1 ~ J6
            joint_keys = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            
            for i, key in enumerate(joint_keys):
                new_val = target_joints[key] + d_q[i]
                # 安全限位
                limits = JOINT_LIMITS[key]
                target_joints[key] = clamp(new_val, limits[0], limits[1])

            # 3. 更新夹爪
            if d_gripper == 1:
                current_gripper += GRIPPER_SPEED
            elif d_gripper == -1:
                current_gripper -= GRIPPER_SPEED
            
            # 夹爪范围限制 0.0 ~ 1.0
            current_gripper = clamp(current_gripper, 0.0, 1.0)

            # 4. 发送指令
            action = {
                f"{k}.pos": v for k, v in target_joints.items()
            }
            action["gripper.pos"] = current_gripper
            
            bot.send_action(action)

            # 5. 打印状态
            print(f"{target_joints['joint_1']:6.2f} | "
                  f"{target_joints['joint_2']:6.2f} | "
                  f"{target_joints['joint_3']:6.2f} | "
                  f"{target_joints['joint_4']:6.2f} | "
                  f"{target_joints['joint_5']:6.2f} | "
                  f"{target_joints['joint_6']:6.2f} | "
                  f"{current_gripper:4.2f} ", end='\r')

            clock.tick(FREQ)

    except KeyboardInterrupt:
        print("\n\n🛑 停止控制...")
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
    finally:
        bot.disconnect()
        pygame.quit()
        print("🔌 机械臂已断开连接")

if __name__ == "__main__":
    main()