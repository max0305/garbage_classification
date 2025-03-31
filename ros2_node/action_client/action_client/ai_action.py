#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from tm12_amm_interfaces.action import Dotask  # 這就是你在 .action 內定義的介面

class AiActionClientNode(Node):
    def __init__(self):
        super().__init__('ai_action_client_node')
        
        # 建立 ActionClient，型別為 Dotask，名稱要跟 Server 相同: "ai_action"
        self._client = ActionClient(self, Dotask, 'ai_action')

        # 等待 Server 出現在 ROS2 網路(上線)後，再送 goal
        self._timer = self.create_timer(1.0, self.send_goal_once_ready)

        # 確保只送一次 goal
        self._goal_sent = False

    def send_goal_once_ready(self):
        if not self._client.server_is_ready():
            self.get_logger().info('等待 ai_action server 上線...')
            return

        if not self._goal_sent:
            self._goal_sent = True
            self.send_goal()

    def send_goal(self):
        goal_msg = Dotask.Goal()
        goal_msg.task = "AI_Action"       # 關鍵字，server 那邊會用它去分派
        goal_msg.scenario = "demo_scene"  # 你自己定義的字串
        goal_msg.repeat_times = 1         # 要抓幾次

        self._client.wait_for_server()
        self.get_logger().info('開始送出 ai_action goal...')
        
        # 真正送出 goal，並指定執行完後的 callback
        self._send_goal_future = self._client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal Rejected :(')
            return
        
        self.get_logger().info('Goal Accepted ! 等待結果中...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        # 如果 action 有定義 feedback，就可在這裡讀取
        self.get_logger().info(f'Receive feedback: {feedback_msg}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result : ok={result.ok},  message={result.result}')

def main(args=None):
    rclpy.init(args=args)
    node = AiActionClientNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
