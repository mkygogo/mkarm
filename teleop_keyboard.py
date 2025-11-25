from follower_mkarm import MKFollower, MKFollowerConfig
import time
import sys
from pynput import keyboard # 导入 pynput 库

# --- 机械臂配置 (来自你的代码) ---
follower_config = MKFollowerConfig(
    port="/dev/ttyACM0",
    joint_velocity_scaling=1.0,
)

print("正在连接机械臂...")
try:
    follower = MKFollower(follower_config)
    follower.connect()
    print("连接成功！")
except Exception as e:
    print(f"连接失败: {e}")
    print("请检查端口 /dev/ttyACM0 是否正确以及机械臂是否已连接。")
    #sys.exit(1) # 连接失败则退出程序


# 保持所有轴当前位置的字典
action = {
    'joint_1.pos': 0.0,
    'joint_2.pos': 0.0,
    'joint_3.pos': 0.0,
    'joint_4.pos': 0.0,
    'joint_5.pos': 0.0,
    'joint_6.pos': 0.0,
    'gripper.pos': 0.0
}

# 将用户输入的数字 (0-6) 映射到 action 字典的键
axis_keys = {
    0: 'joint_1.pos',
    1: 'joint_2.pos',
    2: 'joint_3.pos',
    3: 'joint_4.pos',
    4: 'joint_5.pos',
    5: 'joint_6.pos',
    6: 'gripper.pos'
}

def get_axis_choice():
    """循环提示用户，直到输入一个 0-6 之间的有效数字。"""
    while True:
        try:
            choice_str = input("which axis (0-6): ")
            axis_choice = int(choice_str)
            if axis_choice in axis_keys:
                return axis_choice
            else:
                print("无效输入。请输入 0 到 6 之间的一个数字。")
        except ValueError:
            print("无效输入。请输入一个数字。")

def adjust_radian(current_rad):
    """使用 pynput 监听上下键来调整弧度值。"""
    print("请使用 [↑] (增加 0.01) / [↓] (减少 0.01) 键来调整。按 [Enter] 确认。")
    
    # 使用 pynput.keyboard.Events 来同步获取按键
    with keyboard.Events() as events:
        while True:
            # 打印当前值，使用 \r 和 end='' 来覆盖当前行
            # {current_rad:<8.2f} 确保输出有固定宽度，防止覆盖不全
            print(f"input the rad: {current_rad:<8.2f}", end='\r', flush=True)

            # 阻塞并等待下一个键盘事件
            event = events.get()

            # 我们只关心按键按下的事件
            if isinstance(event, keyboard.Events.Press):
                if event.key == keyboard.Key.up:
                    current_rad += 0.01
                elif event.key == keyboard.Key.down:
                    current_rad -= 0.01
                elif event.key == keyboard.Key.enter:
                    print() # 按下回车后换行
                    break # 退出按键监听循环
            
            # 修正浮点数精度问题
            current_rad = round(current_rad, 2)
            
    return current_rad

# --- 主循环 ---
try:
    while True:
        # 1. 提示用户选择轴
        axis_choice = get_axis_choice()
        key_to_modify = axis_keys[axis_choice]
        
        # 2. 获取该轴的当前弧度值
        current_rad = action[key_to_modify]
        
        # 3. 提示用户调整弧度
        new_rad = adjust_radian(current_rad)
        
        # 4. 更新 action 字典
        action[key_to_modify] = new_rad
        
        # 5. 发送指令给机械臂
        print(f"正在发送: {action}")
        follower.send_action(action)
        print("发送完成。\n")

except KeyboardInterrupt:
    print("\n检测到 Ctrl+C，正在停止程序...")
finally:
    # 确保程序退出时断开连接
    print("断开机械臂连接。")
    follower.disconnect()
    print("程序已退出。")