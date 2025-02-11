import numpy as np
import rclpy
from rclpy.node import Node
from tm_msgs.srv import AskItem

class TMRobotController(Node):
    def __init__(self):
        super().__init__('tm_robot_controller')
        self.cli = self.create_client(AskItem, 'tm_driver/ask_item')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 TM Robot 服務...')
        self.req = AskItem.Request()

    def send_request(self, script_command):
        self.req.item = 'SendScript'
        self.req.id = 'script'
        self.req.script = script_command
        self.future = self.cli.call_async(self.req)

def transform_coordinates(camera_coords):
    """
    將相機座標 (X_cam, Y_cam, Z_cam) 轉換為機械手臂座標系
    """
    X_cam, Y_cam, Z_cam = camera_coords

    # 轉換矩陣 (相機座標系 -> 機械手臂座標系)
    T_cam_to_robot = np.array([
        [1, 0, 0, 0.1],  # X 方向偏移
        [0, 1, 0, -0.05],  # Y 方向偏移
        [0, 0, 1, 0.2],  # Z 方向偏移
        [0, 0, 0, 1]
    ])
    
    P_cam = np.array([X_cam, Y_cam, Z_cam, 1])
    P_robot = np.dot(T_cam_to_robot, P_cam)[:3]  # 機械手臂座標

    return P_robot

def get_place_position(object_class):
    """
    根據物件類別決定放置位置
    """
    place_positions = {
        "bottle": [],
        "can": [],
        "box": [], 
        "carton" : []
    }
    return place_positions.get(object_class, [0.4, 0.0, 0.3])  # 預設放置位置

def move_robot(self, position):
    """
    控制機械手臂移動到指定位置
    """
    call_tm_driver_set_position(self, 2, position, 2, 0.2, 10, True)

def execute_grasp(camera_coords, object_class):
    """
    執行夾取動作
    """
    rclpy.init()
    tm_robot = TMRobotController()

    # 座標轉換
    P_robot = transform_coordinates(camera_coords)

    # 夾爪補償 (夾爪相對手臂偏移)
    T_robot_to_gripper = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -0.1],  # 夾爪相對手臂向下 10cm
        [0, 0, 0, 1]
    ])
    P_gripper = np.dot(T_robot_to_gripper, np.append(P_robot, 1))[:3]

    # 夾取前預備位置
    pre_grasp_position = [P_gripper[0], P_gripper[1], P_gripper[2] + 0.1]  
    grasp_position = P_gripper

    # 取得放置位置
    place_position = get_place_position(object_class)

    # 初始位置
    home_position = [0.0, -0.5, 0.3]

    # **1. 移動到夾取前預備位置**
    move_robot(tm_robot, pre_grasp_position)

    # **2. 移動到夾取位置**
    move_robot(tm_robot, grasp_position)

    # **3. 夾取動作**
    call_grpr2f85_set(tm_robot, 255, 255, 255, wait_time=0)

    # **4. 抬升**
    move_robot(tm_robot, pre_grasp_position)

    # **5. 移動到放置位置**
    move_robot(tm_robot, place_position)

    # **6. 放開夾爪**
    call_grpr2f85_set(tm_robot, 0, 255, 255, wait_time=0)

    # **7. 回到初始位置**
    move_robot(tm_robot, home_position)

    # 關閉 ROS 2 節點
    tm_robot.destroy_node()
    rclpy.shutdown()

# 測試函數
execute_grasp(camera_coords=(0.2, 0.1, 0.3), object_class="bottle")
