import serial
import time 
import numpy as np
import re

class leader_arm:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200):
        # 初始化参数
        self.SERIAL_PORT = serial_port
        self.BAUDRATE = baudrate
        self.ser = None
        
        # 初始化位置数组
        self.arm_pos = [0.0] * 7
        self.angle_pos = [0.0] * 7
        self.zero_angles = [0.0] * 7
        
        # PWM参数
        self.pwm_min = 500
        self.pwm_max = 2500
        self.angle_range = 270

    def connect(self):
        """连接串口"""
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.01)
            print("Serial port opened successfully")
            return True
        except Exception as e:
            print(f"Failed to open serial port: {e}")
            return False

    def disconnect(self):
        """断开串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")

    def send_command(self, cmd):
        """发送命令到串口并返回响应"""
        if not self.ser or not self.ser.is_open:
            print("Serial port is not open")
            return ""
            
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.01)
        response = self.ser.read_all()
        return response.decode('ascii', errors='ignore')

    def pwm_to_angle(self, response_str):
        """将PWM响应转换为角度"""
        match = re.search(r'P(\d{4})', response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = self.pwm_max - self.pwm_min
        angle = (pwm_val - self.pwm_min) / pwm_span * self.angle_range
        return angle

    def angle_to_gripper(self, angle_deg, angle_range=270, pos_min=50, pos_max=730):
        """
        映射伺服角度到夹爪位置
        
        参数:
        - angle_deg: 伺服角度（度）
        - angle_range: 最大伺服角度（默认270°）
        - pos_min: 夹爪关闭位置（默认50）
        - pos_max: 夹爪打开位置（默认730）
        
        返回:
        - 夹爪位置（整数）
        """
        ratio = (angle_deg / angle_range) * 3
        position = pos_min + (pos_max - pos_min) * ratio
        return int(np.clip(position, pos_min, pos_max))

    def release_all_torque(self):
        """释放所有关节的扭矩"""
        if not self.ser or not self.ser.is_open:
            print("Serial port is not open")
            return False
            
        for i in range(7):
            cmd = f'#00{i}PULK!'
            response = self.send_command(cmd)
            print(f"Servo {i} torque released: {response.strip()}")
        return True

    def get_version(self, index=2):
        """获取版本信息"""
        response = self.send_command(f'#00{index}PVER!')
        print(f"Version response: {response.strip()}")
        return response

    def _init_servos(self):
        self.send_command('#000PVER!')
        for i in range(7):  # Only initialize valid servos
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i] = angle if angle is not None else 0.0
            print("Servo initial angle calibration completed")
            

    def update_joint_positions(self):
        """更新所有关节的位置"""
        for i in range(7):
            cmd = f'#00{i}PRAD!'
            response = self.send_command(cmd)
            angle = self.pwm_to_angle(response.strip())
            self.angle_pos[i] = angle if angle is not None else 0.0
            
            if angle is not None:
                angle_offset = (angle - self.zero_angles[i]) 
                angle_rad = np.radians(angle_offset)
                self.arm_pos[i] = angle_rad
            else:
                self.arm_pos[i] = 0.0

    def get_joint_angles_deg(self):
        """获取关节角度（度）"""
        return self.angle_pos.copy()

    def get_joint_angles_rad(self):
        """获取关节角度（弧度）"""
        return self.arm_pos.copy()

    def start_monitoring(self):
        """开始监控机械臂位置（主循环）"""
        if not self.connect():
            return
            
        # 释放所有扭矩
        self.release_all_torque()
        
        # 获取版本信息
        self.get_version()
        
        self._init_servos()

        try:
            while True:
                self.update_joint_positions()
                #print(f"Joint angles (deg): {[f'{angle:.2f}' for angle in self.angle_pos]}")
                print(f"Joint angles (rad): {[f'{arm:.2f}' for arm in self.arm_pos]}")
                time.sleep(0.1)  # 控制更新频率
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        finally:
            self.disconnect()


# 使用示例
if __name__ == "__main__":
    arm = leader_arm()
    arm.start_monitoring()