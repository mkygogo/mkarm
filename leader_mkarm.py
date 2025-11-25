#   Copyright 2025 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.
#   (Adaptation for MKMotorsBus)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from dataclasses import dataclass
import logging
import time
import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorNormMode

from mk_motors import MKMotorsBus

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("dk1_leader_mk")
@dataclass
class MKLeaderConfig(TeleoperatorConfig):
    """
    用于 leader_arm0.py 协议的 MKMotorsBus Leader 臂的配置。
    """
    port: str
    baudrate: int = 115200


class MKLeader(Teleoperator):
    """
    一个 lerobot Teleoperator，用于基于 leader_arm0.py 协议的
    自定义 MKMotorsBus 示教臂。
    
    此示教臂主要用于读取关节位置，并通过 'disable_torque' 释放舵机。
    """
    config_class = MKLeaderConfig
    name = "dk1_leader_mk"

    def __init__(self, config: MKLeaderConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化 MKMotorsBus
        # 注意：MKMotorsBus 在 __init__ 期间执行连接和握手
        self.bus = MKMotorsBus(
            port=self.config.port,
            baudrate=self.config.baudrate,
            motors={
                # --- 修正 ---
                # MotorNormMode.RADIANS 不存在。
                # 我们使用 MotorNormMode.DEGREES (它存在)
                # 并在 get_action 中手动转换为弧度。
                "joint_1": Motor(0, "mk_arm", MotorNormMode.DEGREES),
                "joint_2": Motor(1, "mk_arm", MotorNormMode.DEGREES),
                "joint_3": Motor(2, "mk_arm", MotorNormMode.DEGREES),
                "joint_4": Motor(3, "mk_arm", MotorNormMode.DEGREES),
                "joint_5": Motor(4, "mk_arm", MotorNormMode.DEGREES),
                "joint_6": Motor(5, "mk_arm", MotorNormMode.DEGREES),
                "gripper": Motor(6, "mk_arm", MotorNormMode.DEGREES),
            },
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        # 我们的 MKMotorsBus 在 __init__ 中连接
        return self.bus.ser and self.bus.ser.is_open

    # def connect(self, calibrate: bool = False) -> None:
    #     # MKMotorsBus 在 __init__ 中自动连接和握手
    #     if not self.is_connected:
    #          raise DeviceNotConnectedError(f"{self} 在初始化期间连接失败。")

    #     # 我们仍然调用 configure 来匹配 leader_Feetech.py 的流程
    #     # (例如，确保扭矩被释放)
    #     self.configure()
        
    #     logger.info(f"{self} connected (on __init__) and configured.")

    def connect(self, calibrate: bool = False) -> None:
            """
            (修正后的 connect 方法)
            连接到 MKMotorsBus 并配置舵机。
            """
            
            # 1. 检查是否 *已经* 连接 (防止重复连接)
            if self.is_connected:
                raise DeviceAlreadyConnectedError(f"{self} already connected")

            # 2. 在此处调用 self.bus.connect() 来 *发起* 连接
            #    (这会调用 MKMotorsBus._connect 和 _handshake)
            try:
                self.bus.connect()
            except Exception as e:
                logger.error(f"MKMotorsBus 连接失败: {e}")
                # 重新抛出 DeviceNotConnectedError 以便上层感知
                raise DeviceNotConnectedError(f"{self} a {self.config.port} 连接失败: {e}") from e

            # 3. 如果连接成功，配置电机 (释放扭矩)
            self.configure()
            
            logger.info(f"{self} connected at {self.config.port}.")

    @property
    def is_calibrated(self) -> bool:
        # 我们的总线在 _handshake (在 __init__ 中调用) 期间读取零点偏移量
        return True

    def calibrate(self) -> None:
        # 标定（读取零点）在 MKMotorsBus 的 _handshake 期间自动完成
        pass

    def configure(self) -> None:
        # MKMotorsBus 的 _handshake [在 __init__ 中调用] 已经释放了扭矩。
        # 为确保安全，我们再次调用它。
        self.bus.disable_torque()
        logger.info(f"{self}: 扭矩已释放 (示教臂模式)。")
        
    def setup_motors(self) -> None:
        # MKMotorsBus 是一个只读总线，不支持设置电机ID
        raise NotImplementedError(f"{self} 不支持 setup_motor。")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        
        # --- 修正 ---
        # 1. 模仿 leader_Feetech.py，读取原始 (raw) PWM 值
        action_raw = self.bus.sync_read(normalize=False, data_name="Present_Position")
        
        action = {}
        
        # 2. 手动将 原始PWM 转换为 以零点为中心 的 弧度
        #    我们使用由 MKMotorsBus 在启动时读取的标定数据
        for motor_name, raw_val in action_raw.items():
            # self.bus.calibration 由 MKMotorsBus.read_calibration() 填充
            calib = self.bus.calibration[motor_name]
            
            # 获取零点PWM (在 _handshake 时读取)
            zero_pwm = calib.homing_offset
            
            # 获取完整的PWM范围 (例如 2500 - 500 = 2000)
            full_pwm_range = calib.range_max - calib.range_min
            
            # 根据 leader_arm0.py，这个PWM范围对应 270 度
            full_rad_range = np.radians(270.0) 
            
            if full_pwm_range == 0:
                # 避免除以零
                relative_rad = 0.0
            else:
                # (raw_val - zero_pwm) 是相对PWM
                # (relative_pwm / full_pwm_range) 是归一化位置 (例如 -0.5 到 0.5)
                # ... * full_rad_range 将其转换为弧度
                relative_rad = (float(raw_val) - float(zero_pwm)) / float(full_pwm_range) * full_rad_range
            
            action[f"{motor_name}.pos"] = relative_rad
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # 示教臂不支持力反馈
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # MKMotorsBus 使用 'close_bus()' 方法
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")



if __name__ == "__main__" :
    # 配置日志记录以查看调试输出
    logging.basicConfig(level=logging.INFO)
    
    # 替换为您的实际串口
    LEADER_PORT = "/dev/ttyUSB0" 
    
    leader_config = MKLeaderConfig(
        port=LEADER_PORT,
        baudrate=115200  # 匹配 leader_arm0.py
    )

    leader = MKLeader(leader_config)
    
    # connect() 会验证连接并配置（释放扭矩）
    leader.connect()

    freq = 10 # Hz
    print(f"开始监控 Leader 臂: {LEADER_PORT} (频率: {freq} Hz)")
    print("按 Ctrl+C 停止。")

    try:
        while True:
            action = leader.get_action()
            
            # 格式化输出以便更易读
            action_str = ", ".join(
                f"{name.split('.')[0]}: {pos:+.3f}" for name, pos in action.items()
            )
            print(f"Action (rad): [ {action_str} ]", end="\r")
            
            time.sleep(1/freq)
            
    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        leader.disconnect()
        print("Leader 臂已断开连接。")