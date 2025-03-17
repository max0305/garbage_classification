#!/usr/bin/env python3

VERSION = "1.0.0"

'''
# 服務端範例
config = ZMQConfig(ip="127.0.0.1", port=5555, timeout=5000)
with zmq_ai_srv_srv(config) as server:
    server.run()

# 客戶端範例
config = ZMQConfig(ip="127.0.0.1", port=5555, timeout=5000)
with zmq_ai_srv_clt(config) as client:
    result = client.run(rgb_image, depth_image, camera_matrix, dist_coeffs)
    if result:
        print(f"收到位姿結果: {result}")
'''

import os
import logging
from datetime import datetime
from math import pi
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import json
import zmq
import cv2
from dataclasses import dataclass
from functools import wraps  # 新增這行
import time
import psutil

# 資料驗證裝飾器
def validate_image(func):
    """驗證影像資料的裝飾器"""
    @wraps(func)
    def wrapper(self, rgb: np.ndarray, depth: np.ndarray, *args, **kwargs):
        if rgb is None or depth is None:
            raise ValueError("影像不能為空")
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB影像必須是三通道影像")
        if len(depth.shape) != 2:
            raise ValueError("深度影像必須是單通道影像")
        return func(self, rgb, depth, *args, **kwargs)
    return wrapper

def validate_camera_params(func):
    """驗證相機參數的裝飾器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 處理位置參數
        if len(args) >= 3:  # rgb, depth, camera_matrix 是位置參數
            camera_matrix = args[2]
        else:
            camera_matrix = kwargs.get('camera_matrix', None)

        if len(args) >= 4:  # dist_coeffs 是第四個位置參數
            dist_coeffs = args[3]
        else:
            dist_coeffs = kwargs.get('dist_coeffs', None)

        # 驗證相機矩陣
        if camera_matrix is not None:
            if not isinstance(camera_matrix, np.ndarray) or camera_matrix.shape != (3, 3):
                raise ValueError("相機矩陣必須是3x3的numpy陣列")

        # 驗證畸變係數
        if dist_coeffs is not None:
            if not isinstance(dist_coeffs, np.ndarray) or len(dist_coeffs) != 5:
                raise ValueError("畸變係數必須是長度為5的numpy陣列")

        return func(self, *args, **kwargs)
    return wrapper

# 性能監控裝飾器
def performance_monitor(func):
    """監控函數執行時間和資源使用的裝飾器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        # 獲取資源使用情況
        process = psutil.Process()
        cpu_percent = psutil.cpu_percent()
        memory_info = process.memory_info()

        logging.info(f"函數 {func.__name__} 執行時間: {execution_time:.2f}秒")
        logging.debug(f"CPU 使用率: {cpu_percent:.2f}%")
        logging.debug(f"記憶體使用: {memory_info.rss / 1024 / 1024:.2f} MB")

        return result
    return wrapper

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class ZMQConfig:
    """ZMQ 設定值"""
    ip: str = "127.0.0.1"
    port: int = 5555
    timeout: int = 30000  # 預設超時時間（毫秒）
    linger: int = 1000    # 設定關閉時的等待時間（毫秒）
    is_server: bool = False  # 是否為服務端

    def __post_init__(self):
        if not isinstance(self.port, int) or self.port <= 0:
            raise ValueError("連接埠必須是正整數")

        # 服務端和客戶端的不同設定
        if self.is_server:
            self.timeout = -1  # 服務端無限等待
        else:
            if self.timeout <= 0:
                self.timeout = 30000  # 客戶端預設 30 秒超時

@dataclass
class CameraParams:
    """相機參數"""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

class ZMQBase:
    """ZMQ 基礎類別"""
    def __init__(self, config: ZMQConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.context = None
        self.socket = None

    def _initialize_zmq(self, socket_type: int):
        """初始化 ZMQ"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(socket_type)
            self.socket.setsockopt(zmq.RCVTIMEO, self.config.timeout)
            self.socket.setsockopt(zmq.SNDTIMEO, self.config.timeout)
            self.socket.setsockopt(zmq.LINGER, self.config.linger)  # 設定 linger
            # 增加重連機制
            self.socket.setsockopt(zmq.RECONNECT_IVL, 1000)  # 1 秒重連間隔
            self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)  # 最大重連間隔 5 秒
        except zmq.ZMQError as e:
            self.logger.error(f"ZMQ 初始化失敗: {e}")
            raise

    def _retry_connection(self, max_retries: int = 3, delay: float = 1.0) -> bool:
        """連線重試機制

        Args:
            max_retries: 最大重試次數
            delay: 重試間隔（秒）

        Returns:
            bool: 是否連線成功
        """
        for attempt in range(max_retries):
            try:
                if self.socket:
                    self.socket.close()
                if self.context:
                    self.context.term()

                self._initialize_zmq(self.socket_type)
                self.logger.info(f"重新連線成功 (嘗試 {attempt + 1}/{max_retries})")
                return True

            except zmq.ZMQError as e:
                self.logger.warning(f"重新連線失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)

        self.logger.error("重新連線失敗，已達最大重試次數")
        return False

    def _heartbeat(self) -> bool:
        """心跳檢測"""
        try:
            if isinstance(self, zmq_ai_srv_srv):
                self.socket.send_json({'type': 'heartbeat'})
                response = self.socket.recv_json()
                return response.get('status') == 'alive'
            elif isinstance(self, zmq_ai_srv_clt):
                response = self.socket.recv_json()
                if response.get('type') == 'heartbeat':
                    self.socket.send_json({'status': 'alive'})
                    return True
            return False
        except Exception as e:
            self.logger.error(f"心跳檢測失敗: {e}")
            return False

    def close(self):
        """關閉連接"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class zmq_ai_srv_srv(ZMQBase):
    """AI 服務端"""
    def __init__(self, ip = "127.0.0.1", port = 5555):
        config = ZMQConfig(ip=ip, port=port, is_server=True)
        super().__init__(config)
        self._initialize_zmq(zmq.REP)
        self.socket.connect(f"tcp://{self.config.ip}:{self.config.port}")
        self.logger.info(f"服務端已連接到 {self.config.ip}:{self.config.port}")

    def __receive_RGBD_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """接收 RGB-D 影像和相機參數"""
        try:
            message = self.socket.recv()
            if len(message) < 64:
                raise ValueError("接收到的訊息太短")

            header = np.frombuffer(message[:64], dtype=np.float32)
            rgb_size, depth_size = int(header[0]), int(header[1])

            if rgb_size <= 0 or depth_size <= 0:
                raise ValueError(f"無效的影像大小: RGB={rgb_size}, Depth={depth_size}")

            camera_matrix = header[2:11].reshape(3,3)
            dist_coeffs = header[11:16]

            # 解析影像資料
            current_pos = 64
            rgb_data = message[current_pos:current_pos+rgb_size]
            depth_data = message[current_pos+rgb_size:current_pos+rgb_size+depth_size]

            # 解碼影像
            rgb = cv2.imdecode(np.frombuffer(rgb_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            depth = cv2.imdecode(np.frombuffer(depth_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            if rgb is None or depth is None:
                raise ValueError("影像解碼失敗")

            return rgb, depth, camera_matrix, dist_coeffs

        except Exception as e:
            self.logger.error(f"接收影像資料時發生錯誤: {e}")
            raise

    @validate_image
    @validate_camera_params
    def inference(self, rgb: np.ndarray, depth: np.ndarray,
                   camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Optional[List[float]]:
        """執行 AI 推論"""
        try:
            # TODO: 實現實際的 AI 推論邏輯
            pose = [0.3571, -0.5795, 0.5, -pi, 0., pi/4]

            return pose

        except Exception as e:
            self.logger.error(f"AI 推論過程發生錯誤: {e}")
            return None

    @performance_monitor
    def run(self):
        """執行服務"""
        self.logger.info("AI 服務開始運行")
        reconnect_delay = 1.0  # 初始重連延遲
        max_reconnect_delay = 30.0  # 最大重連延遲

        while True:
            try:
                # 接收影像
                rgb, depth, camera_matrix, dist_coeffs = self.__receive_RGBD_image()

                # 執行推論
                result = self.inference(rgb, depth, camera_matrix, dist_coeffs)

                # 儲存推論資料
                if result is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    data = {
                        'rgb': rgb,
                        'depth': depth,
                        'camera_matrix': camera_matrix,
                        'dist_coeffs': dist_coeffs,
                        'pose': result
                    }
                    relative_path = os.path.join("inference_data", f"inference_{timestamp}")
                    self.save_data(data, relative_path)

                # 發送回應
                self.__send_response(result)

                # 重置重連延遲
                reconnect_delay = 1.0

            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    self.logger.warning("等待連接超時，嘗試重新連接...")
                    time.sleep(reconnect_delay)
                    # 指數退避重連延遲
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue
                else:
                    self.logger.error(f"ZMQ 通訊錯誤: {e}")
                    break

            except Exception as e:
                self.logger.error(f"處理過程發生錯誤: {e}")
                self.__send_response(None)
                time.sleep(1)  # 避免過快重試
    def save_data(self, data: Dict[str, Any], filepath: str) -> None:
        """儲存推論資料

        Args:
            data: 包含影像和推論結果的字典
            filepath: 儲存路徑 (相對於專案根目錄)
        """
        try:
            # 取得專案根目錄路徑
            root_dir = os.path.dirname(os.path.abspath(__file__))

            # 建立完整路徑
            full_path = os.path.join(root_dir, filepath)

            # 確保目錄存在
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # 儲存影像，加入錯誤檢查
            if not cv2.imwrite(f"{full_path}_rgb.jpg", data['rgb']):
                raise IOError("RGB 影像儲存失敗")
            if not cv2.imwrite(f"{full_path}_depth.jpg", data['depth']):
                raise IOError("深度影像儲存失敗")

            # 儲存其他資料
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'camera_matrix': data['camera_matrix'].tolist(),
                'dist_coeffs': data['dist_coeffs'].tolist(),
                'pose': data['pose']
            }

            # 使用安全的檔案寫入方式
            metadata_path = f"{full_path}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            self.logger.info(f"已儲存推論資料至 {full_path}")

        except IOError as e:
            self.logger.error(f"檔案 I/O 錯誤: {e}")
        except Exception as e:
            self.logger.error(f"儲存資料時發生錯誤: {str(e)}")
            self.logger.debug("錯誤詳情", exc_info=True)

    def __send_response(self, result: Optional[List[float]]) -> None:
        """發送推論結果回應

        Args:
            result: 位姿結果列表或 None（發生錯誤時）
        """
        try:
            if result is None:
                # 發送錯誤回應
                self.socket.send_json({'status': 'error', 'data': None})
            else:
                # 發送成功回應
                self.socket.send_json({'status': 'success', 'data': result})
        except Exception as e:
            self.logger.error(f"發送回應時發生錯誤: {e}")
            # 嘗試發送錯誤訊息
            try:
                self.socket.send_json({'status': 'error', 'data': None})
            except:
                pass

class zmq_ai_srv_clt(ZMQBase):
    """AI 客戶端"""
    def __init__(self, ip = "127.0.0.1", port = 5555):
        config = ZMQConfig(ip=ip, port=port)
        super().__init__(config)
        self._initialize_zmq(zmq.REQ)
        self.socket.bind(f"tcp://{self.config.ip}:{self.config.port}")
        self.logger.info(f"客戶端已綁定到 {self.config.ip}:{self.config.port}")

    @validate_image
    @validate_camera_params
    def run(self, rgb: np.ndarray, depth: np.ndarray,
            camera_matrix: np.ndarray = np.eye(3),
            dist_coeffs: np.ndarray = np.zeros(5)) -> Optional[List[float]]:
        """執行一次請求"""
        try:
            message = self.__prepare_message(rgb, depth, camera_matrix, dist_coeffs)
            self.socket.send(message)
            return self.__receive_response()
        except Exception as e:
            self.logger.error(f"執行請求時發生錯誤: {e}")
            return None

    @validate_image
    def __prepare_message(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> bytes:
        """準備傳送訊息"""
        try:
            # 影像壓縮
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
            _, rgb_encoded = cv2.imencode('.jpg', rgb, encode_params)
            _, depth_encoded = cv2.imencode('.jpg', depth, encode_params)

            # 準備標頭
            header = np.zeros(16, dtype=np.float32)
            header[0] = len(rgb_encoded)
            header[1] = len(depth_encoded)
            header[2:11] = camera_matrix.flatten()
            header[11:16] = dist_coeffs

            return header.tobytes() + rgb_encoded.tobytes() + depth_encoded.tobytes()
        except Exception as e:
            self.logger.error(f"準備訊息時發生錯誤: {e}")
            raise

    def __receive_response(self) -> Optional[List[float]]:
        """接收伺服器回應

        Returns:
            位姿結果列表或 None（發生錯誤時）
        """
        try:
            response = self.socket.recv_json()
            if response['status'] == 'success':
                return response['data']
            else:
                self.logger.error("伺服器回傳錯誤狀態")
                return None
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                self.logger.error("等待回應超時")
            else:
                self.logger.error(f"接收回應時發生 ZMQ 錯誤: {e}")
            return None
        except Exception as e:
            self.logger.error(f"接收回應時發生錯誤: {e}")
            return None

class model_zmq_srv(zmq_ai_srv_srv):
    def __init__(self, ip = "127.0.0.1", port = 5555):
        super().__init__(ip = ip, port = port)

    def inference(self, rgb, depth, camera_matrix, dist_coeffs):
        # 這個方法會被父類別的 run() 使用
        try:
            # 您的自定義推論邏輯
            pose = [0.1, 0.2, 0.3, 0, 0, 0]  # 自定義位姿
            self.logger.info(f"demo 成功")
            return pose
        except Exception as e:
            self.logger.error(f"自定義推論過程發生錯誤: {e}")
            return None

import unittest
import numpy as np
from .zmq_ai_service import ZMQConfig, zmq_ai_srv_clt, model_zmq_srv

class TestZMQAIService(unittest.TestCase):
    def setUp(self):
        # 建立測試影像
        self.ip = "127.0.0.1"
        self.port = 5555
        self.rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.depth = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)  # 改為 uint16
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros(5)

    def test_server_client_communication(self):
        # 建立伺服器執行緒
        import threading
        server_ready = threading.Event()

        def run_server():
            with model_zmq_srv(ip = self.ip, port = self.port) as server:
                server_ready.set()  # 通知伺服器準備就緒
                server.run()

        # 在背景執行伺服器
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True  # 設定為背景執行緒
        server_thread.start()

        # 等待伺服器準備就緒
        server_ready.wait(timeout=5)

        # 執行客戶端測試
        try:
            with zmq_ai_srv_clt(ip = self.ip, port = self.port) as client:
                result = client.run(self.rgb, self.depth,
                                  self.camera_matrix, self.dist_coeffs)
                self.assertIsNotNone(result, "結果不應為 None")
                self.assertEqual(len(result), 6, "結果應該包含 6 個元素")
        finally:
            # 清理資源
            server_thread.join(timeout=1)

    def test_invalid_image_format(self):
        """測試無效的影像格式"""
        with zmq_ai_srv_clt(ip = self.ip, port = self.port) as client:
            invalid_rgb = np.random.randint(0, 255, (480, 640), dtype=np.uint8)  # 錯誤的維度
            with self.assertRaises(ValueError):
                client.run(invalid_rgb, self.depth, self.camera_matrix, self.dist_coeffs)

    def test_invalid_camera_params(self):
        """測試無效的相機參數"""
        with zmq_ai_srv_clt(ip = self.ip, port = self.port) as client:
            # 測試無效的相機矩陣
            invalid_matrix = np.eye(2)  # 2x2 矩陣
            with self.assertRaises(ValueError):
                client.run(self.rgb, self.depth, invalid_matrix, self.dist_coeffs)

            # 測試無效的畸變係數
            invalid_dist = np.zeros(3)  # 只有3個係數
            with self.assertRaises(ValueError):
                client.run(self.rgb, self.depth, self.camera_matrix, invalid_dist)

            # 測試錯誤類型
            invalid_type = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 列表而不是 numpy 陣列
            with self.assertRaises(ValueError):
                client.run(self.rgb, self.depth, invalid_type, self.dist_coeffs)

if __name__ == '__main__':
    unittest.main()
