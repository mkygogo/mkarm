# mk_motors.py

import logging
import re
import time
from copy import deepcopy

import serial

# --- 修正: 从 'motors_bus' 模块正确导入 ---
from lerobot.motors.motors_bus import (
    Motor,
    MotorCalibration,
    MotorsBus,
    NameOrID,
    Value,
)


logger = logging.getLogger("mk_motors") # 为日志记录器指定一个名称

# --- 定义 "mk_arm" 模型的控制表 ---
MK_ARM_CONTROL_TABLE = {
    "Torque_Enable": (40, 1),
    "Present_Position": (56, 2),
}

MODEL_CONTROL_TABLE = {"mk_arm": MK_ARM_CONTROL_TABLE}
MODEL_RESOLUTION = {"mk_arm": 2000} # 范围 (500-2500)
NORMALIZED_DATA = ["Present_Position"]
MODEL_BAUDRATE_TABLE = {"mk_arm": {}}
MODEL_ENCODING_TABLE = {"mk_arm": {}}
MODEL_NUMBER_TABLE = {"mk_arm": 0}
MODEL_PROTOCOL = {"mk_arm": 99} # 自定义


class MKMotorsBus(MotorsBus):
    """
    一个 lerobot MotorsBus 接口，用于通过自定义ASCII串口协议
    (如 leader_arm0.py 中所示) 控制的6+1轴机械臂。
    """

    # --- 映射到基类的类变量 ---
    apply_drive_mode = False
    available_baudrates = [115200, 57600, 19200]
    default_baudrate = 115200
    default_timeout = 1000
    
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)


    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        baudrate: int = 115200,
    ):
        # ***重要***: 我们必须在 'super().__init__' 之前初始化 'self.ser'
        # 因为基类 'connect()' (在 super().__init__ 中调用) 
        # 会检查 'self.is_connected'，而 'self.is_connected' 依赖 'self.ser'
        self.ser = None
        
        self.baudrate = baudrate
        self.pwm_min = 500
        self.pwm_max = 2500
        self.angle_range = 270
        
        # 调用基类构造函数
        super().__init__(port, motors, calibration)

        # 存储标定的零点 *PWM* 值
        max_id = max(self.ids) if self.ids else 0
        self.zero_angles_pwm = [0.0] * (max_id + 1)

        # --- 连接逻辑从 __init__ 移至 connect() ---
        # (基类的 __init__ 会自动调用 connect())

    # --- 核心通信方法 ---

    # def _connect(self) -> None:
    #     """(由基类调用) 实际的连接逻辑。"""
    #     try:
    #         self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
    #         logger.info(f"MKMotorsBus: 成功打开串口 {self.port} at {self.baudrate}")
    #     except Exception as e:
    #         logger.error(f"MKMotorsBus: 无法打开串口 {self.port}: {e}")
    #         raise
        
    #     # 连接后立即执行握手
    #     self._handshake()

    def _connect(self, handshake: bool) -> None:
            """(由基类调用) 实际的连接逻辑。"""
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
                logger.info(f"MKMotorsBus: 成功打开串口 {self.port} at {self.baudrate}")
            except Exception as e:
                logger.error(f"MKMotorsBus: 无法打开串口 {self.port}: {e}")
                raise
            
            # 仅当基类请求握手时 (默认情况) 才执行握手
            if handshake:
                self._handshake()
            else:
                logger.info("MKMotorsBus: 已连接 (跳过握手)。")

    def _disconnect(self) -> None:
        """(由基类调用) 实际的断开连接逻辑。"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info(f"MKMotorsBus: 串口 {self.port} 已关闭")
        self.ser = None

    def set_timeout(self, timeout_ms: int | None = None) -> None:
        """
        (覆盖基类)
        基类 'MotorsBus' 假定有一个 'port_handler' 
        (用于 Dynamixel SDK)。
        我们的类使用 'pyserial' (self.ser)，它没有 'port_handler'。
        
        我们的超时已经在 _connect 方法中为 self.ser 设置了 (timeout=0.01)，
        所以我们覆盖此方法，使其不执行任何操作 (No-op)
        以避免 'AttributeError'。
        """
        # 不执行任何操作 (No-op)
        pass
    
    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        num_retry: int = 0,
        normalize: bool = False,
        raise_on_error: bool = False,
    ) -> dict[str, Value]:
        """
        (覆盖基类)
        基类的 'sync_read' 依赖 'self.sync_reader'，
        这与我们的 ASCII 协议不兼容。
        
        我们实现一个精简版本，它直接调用我们的 '_read_from_bus'，
        并手动处理 motor_name -> id 的转换和归一化。
        """
        motor_names = self._get_motors_list(motors)
        motor_ids = [self.motors[name].id for name in motor_names]

        # 1. 直接调用我们的ASCII读取方法
        #    (这会返回 {id: raw_pwm_value})
        ids_values = self._read_from_bus(data_name, motor_ids, num_retry=num_retry)

        # 2. 将 {id: value} 转换回 {name: value}
        names_values = {}
        for name in motor_names:
            id_ = self.motors[name].id
            names_values[name] = ids_values.get(id_, 0) # 使用 .get 避免 KeyErrors

        # 3. 处理归一化 (如果 'normalize=True')
        if normalize:
            names_values = self._normalize(data_name, names_values)
        
        return names_values

    @property
    def is_connected(self) -> bool:
        """(由基类使用) 检查连接状态。"""
        return self.ser and self.ser.is_open

    def _send_command(self, cmd: str) -> str:
        """发送ASCII命令并读取响应。"""
        if not self.is_connected:
            logger.error("MKMotorsBus: 串口未打开")
            return ""
            
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.01) # 等待响应
        response = self.ser.read_all()
        return response.decode('ascii', errors='ignore')

    def _handshake(self) -> None:
        """
        验证与机械臂的通信并获取初始零点位置。
        """
        logger.info("MKMotorsBus: 正在执行握手...")
        
        # 1. 释放所有扭矩 (来自 leader_arm.release_all_torque)
        #    (我们在这里直接调用 _disable_torque 而不是 self.disable_torque
        #     以避免基类中复杂的锁/重试逻辑)
        logger.info("MKMotorsBus: 释放所有电机扭矩...")
        for id_ in self.ids:
            self._disable_torque(id_, "mk_arm") # 使用内部方法

        # 2. 获取版本信息
        try:
            cmd = '#000PVER!'
            response = self._send_command(cmd)
            if not response:
                raise ConnectionError("无响应。请检查波特率和连接。")
            logger.info(f"MKMotorsBus: 版本响应: {response.strip()}")
        except Exception as e:
            logger.error(f"MKMotorsBus: 握手失败: {e}")
            self._disconnect() # 握手失败时关闭
            raise

        # 3. 标定零点角度 (读取初始PWM值)
        logger.info("MKMotorsBus: 正在标定零点位置 (读取初始PWM值)...")
        errors = 0
        for id_ in self.ids:
            cmd = f'#{id_:03d}PRAD!'
            response = self._send_command(cmd)
            match = re.search(r'P(\d{4})', response.strip())
            if match:
                pwm_val = int(match.group(1))
                self.zero_angles_pwm[id_] = pwm_val
            else:
                logger.warning(f"MKMotorsBus: 无法读取舵机 {id_} 的初始位置")
                errors += 1
        
        if errors == 0:
            logger.info("MKMotorsBus: 所有舵机的初始零点标定完成。")
        else:
            logger.warning("MKMotorsBus: 部分舵机标定失败。")
            
        # 4. 读取标定数据并缓存
        self.read_calibration()

    def _read_from_bus(
        self, data_name: str, ids: list[int], num_retry: int = 0
    ) -> dict[int, int]:
        """(由基类调用) 从总线同步读取数据。"""
        if data_name != "Present_Position":
            raise NotImplementedError(f"MKMotorsBus: 不支持读取 '{data_name}'。")
            
        results = {}
        for id_ in ids:
            cmd = f'#{id_:03d}PRAD!'
            response = ""
            
            for _ in range(num_retry + 1):
                response = self._send_command(cmd)
                match = re.search(r'P(\d{4})', response.strip())
                if match:
                    results[id_] = int(match.group(1))
                    break
            else:
                logger.warning(f"MKMotorsBus: 读取舵机 {id_} 失败 (响应: '{response.strip()}')")
                results[id_] = 0
        
        return results

    def _write_to_bus(
        self, data_name: str, ids_values: dict[int, int], num_retry: int = 0
    ) -> None:
        """(由基类调用) 向总线同步写入数据。"""
        if data_name == "Torque_Enable":
            for id_, value in ids_values.items():
                if value == 0:  # 禁用扭矩 (释放)
                    self._disable_torque(id_, "mk_arm", num_retry)
                else:
                    logger.warning(f"MKMotorsBus: 不支持为舵机 {id_} 启用扭矩。")
        
        elif data_name == "Goal_Position":
            raise NotImplementedError("MKMotorsBus: 不支持写入 'Goal_Position'。")
        else:
            raise NotImplementedError(f"MKMotorsBus: 不支持写入 '{data_name}'。")

    # --- 实现缺失的抽象方法 ---
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        (修正) (抽象方法实现) 释放一个或多个电机的扭矩。
        我们绕过基类的 'self.write()' (因为它用于字节协议)
        并直接调用我们包含 ASCII 逻辑的 '_disable_torque'。
        """
        motor_names = self._get_motors_list(motors)
        for motor_name in motor_names:
            # 从 'motors' 字典中获取 ID 和 model
            motor_id = self.motors[motor_name].id
            model = self.motors[motor_name].model
            
            # 直接调用我们自己的 _disable_torque，它包含 'PULK!' 命令
            self._disable_torque(motor_id, model, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """
        (修正) (抽象方法实现) (不支持) 启用一个或多个电机的扭矩。
        我们绕过 'self.write()'，只记录一个警告。
        """
        motor_names = self._get_motors_list(motors)
        ids = [self.motors[m].id for m in motor_names]
        logger.warning(
            f"MKMotorsBus: 试图为舵机 {ids} 启用扭矩，但此协议不支持该操作。"
        )
        # 不执行任何操作 (No-op)，因为我们没有启用扭矩的命令
        # 并且我们必须避免调用 self.write()
        pass

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        """(抽象方法实现)"""
        # 我们的自定义协议总是兼容的
        pass

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """(抽象方法实现)"""
        # 我们的协议不使用符号编码
        return ids_values

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        """(抽象方法实现) 禁用单个电机的扭矩。"""
        cmd = f'#{motor_id:03d}PULK!'
        for _ in range(num_retry + 1):
            self._send_command(cmd)
            # (我们无法验证写入是否成功)

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """(抽象方法实现)"""
        # 我们的协议不使用符号编码
        return ids_values

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        """(抽象方法实现)"""
        raise NotImplementedError("MKMotorsBus: 不支持扫描单个电机。")

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """(抽象方法实现)"""
        # 我们的协议使用ASCII，不使用字节块
        raise NotImplementedError("MKMotorsBus: 不使用字节块。")

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """(抽象方法实现)"""
        # 我们的协议没有广播 ping
        logger.warning("MKMotorsBus: 不支持 'broadcast_ping'。")
        return None

    def configure_motors(self, **kwargs) -> None:
        """(抽象方法实现)"""
        # 我们的配置在 _handshake 中完成
        logger.debug("MKMotorsBus: 'configure_motors' 被调用，无操作。")
        pass

    @property
    def is_calibrated(self) -> bool:
        """(抽象方法实现)"""
        # 我们假设在 _handshake 之后即已标定
        return self.calibration is not None

    # --- 覆盖已有的非抽象方法 (根据需要) ---

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """
        返回总线上电机的当前标定状态。
        """
        calibration = {}
        for motor_name, motor in self.motors.items():
            id_ = motor.id
            calibration[motor_name] = MotorCalibration(
                id=id_,
                drive_mode=0,
                homing_offset=int(self.zero_angles_pwm[id_]), # 零点PWM
                range_min=self.pwm_min, # 500
                range_max=self.pwm_max, # 2500
            )
        
        # 缓存标定 (重要！)
        self.calibration = calibration
        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """(覆盖基类) 向电机写入标定数据。"""
        raise NotImplementedError("MKMotorsBus: 不支持写入永久性标定数据。")

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        计算相对于PWM范围中间点的归位偏移量。
        """
        half_turn_homings = {}
        center_pwm = self.pwm_min + (self.pwm_max - self.pwm_min) / 2.0  # 1500.0
        
        for motor, pos in positions.items():
            half_turn_homings[motor] = pos - int(center_pwm)
            
        return half_turn_homings