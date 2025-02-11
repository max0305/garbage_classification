import sys
import numpy as np                                     # Python数值计算库
import cv2                                             # Opencv图像处理库
import time
import threading
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, \
                         CancelResponse, \
                         GoalResponse

from cv_bridge import CvBridge                         # ROS与OpenCV图像转换类

from std_msgs.msg import Header
from sensor_msgs.msg import Image                      # 图像消息类型
from ai_interfaces.action import Simple
from tm_msgs.srv import SetPositions
from grpr2f85_ifaces.srv import SetGripperState #, \
                                #Reset, \
                                #GetGripperStatus
from realsense2_camera_msgs.msg import RGBD

from algorithm import my_algorithm

class Ai_controller(Node):
    def __init__(self, name='ai_controller'):
        super().__init__(name)

        self.cv_bridge = CvBridge()

        self.act_goal = Simple.Goal()
        self.act_result = Simple.Result()
        self.act_feedback = Simple.Feedback()
        
        self.tm12_set_position_req = SetPositions.Request()
        self.tm12_set_position_resp = SetPositions.Response()

        self.grpr2f85_set_req = SetGripperState.Request()
        self.grpr2f85_set_resp = SetGripperState.Response()

        self.realsens_data = RGBD()

        self.algo = my_algorithm.simple_algorithm(task='fixed', scenario='straight_up_down')

        # Initialize the action server
        self._action_server = ActionServer(
            self,
            Simple,
            'ai_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
            )
        
        # Create clients for both AddTwoInts and MultiplyTwoInts services
        self.tm12_client = self.create_client(
            SetPositions, 
            'set_position'
            )
        
        self.gripper2f85_client = self.create_client(
            SetGripperState, 
            'set_gripper_state'
            )
        
        self.realsense_subscriber = self.create_subscription(
            RGBD, 
            'rgbd', 
            self.realsense_callback,
            10
            )
        
        while not self.tm12_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for TM12 service ...')
        while not self.gripper2f85_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for 2f85 service ...')

    def goal_callback(self, goal_request):
        """Accept all goals for simplicity."""
        self.get_logger().info('Received goal request.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Allow cancellations."""
        self.get_logger().info('Received cancel request.')
        return CancelResponse.ACCEPT

    ## ######################################
    def realsense_callback(self):
        return 1
    
    async def call_tm_driver_set_position(self, 
                                          positions,
                                           ): #default as home position
        self.get_logger().info(f'Calling tm_driver set_position service')
        self.tm12_set_position_req.motion_type = self.algo.get_setting()['motion_type']
        self.tm12_set_position_req.positions = positions
        self.tm12_set_position_req.velocity = self.algo.get_setting()['velocity']
        self.tm12_set_position_req.acc_time = self.algo.get_setting()['acc_time']
        self.tm12_set_position_req.blend_percentage = self.algo.get_setting()['blend_percentage']
        self.tm12_set_position_req.fine_goal =self.algo.get_setting()['fine_goal']

        future = self.tm12_client.call_async(self.tm12_set_position_req)
        await future
        return future.result()

    async def call_my_2f85gripper_set(self, position=0, speed=255, force=255, wait_time=0):
        self.get_logger().info(f'Calling my_2f85gripper_set service')
        self.grpr2f85_set_req.position = position
        self.grpr2f85_set_req.speed = speed
        self.grpr2f85_set_req.force = force
        self.grpr2f85_set_req.wait_time = wait_time

        future = self.gripper2f85_client.call_async(self.grpr2f85_set_req)
        await future
        return future.result()
        
    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        #timer t_0
        '''
        Sensor: sample scene to image, 
        To a func
        ''' 
        sample_trajectory = self.algo.get_sample_trajectory()
        for state in sample_trajectory:
            result = await self.call_tm_driver_set_position(
                positions = state
            )
            self.act_feedback.current_state = f'Take picture at Pose: {state}'
            # take picture
            goal_handle.publish_feedback(self.act_feedback)


        #timer t_1
        '''
        algorithm: image to trajectory
        '''

        result_trajectory = self.algo.forward()
        
        
        #timer t_2
        '''
        manipulation: trajectory to command
        (Task, Scenario) -> {(Operator, Command)_i}_0^N
        To a func
        '''
        for state in result_trajectory:
            result = await self.call_tm_driver_set_position(
                positions = state[0:6]
            )
            result = await self.call_my_2f85gripper_set(
                positions = state[6]
            )
            self.act_feedback.current_state = f'Pose {state}'
            goal_handle.publish_feedback(self.act_feedback)

        goal_handle.succeed()
        #timer t_3

        result = Simple.Result()
        result.ok = True
        result.result = f'Job complete'
        return result

#################################################################################


def main(args=None):
    rclpy.init(args=args)
    node = Ai_controller()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



