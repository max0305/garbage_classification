#!/usr/bin/env python3

import sys
import os
import numpy as np                                     # Python数值计算库
import cv2                                             # Opencv图像处理库
import time
import threading
import math
from math import pi
from typing import TypedDict, Optional
import numpy.typing as npt
import signal
import logging
import asyncio

import pyrealsense2 as rs
from ultralytics import YOLO
# from my_zmq_py.zmq_ai_service import zmq_ai_srv_clt



import rclpy
import rclpy.callback_groups
from rclpy.task import Future
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, \
								ReentrantCallbackGroup
from rclpy.action import ActionServer, \
				CancelResponse, \
				GoalResponse
from rclpy.action.server import ServerGoalHandle


from cv_bridge import CvBridge                         # ROS与OpenCV图像转换类


from rcl_interfaces.msg import ParameterDescriptor, \
				SetParametersResult
from std_msgs.msg import Header, Char                   # ROS消息类型
from geometry_msgs.msg import Twist, Pose, Quaternion   # ROS消息类型
from sensor_msgs.msg import Image
from realsense2_camera_msgs.msg import RGBD
from tm_msgs.msg import FeedbackState

from std_srvs.srv import Trigger
from tm_msgs.srv import SetPositions
from grpr2f85_ifaces.srv import SetGripperState, GetGripperStatus

from tm12_amm_interfaces.action import Dotask, Calibration

# from tm12_amm_interfaces.srv import AiAction
class CalibrationResult(TypedDict):
    error: Optional[float]
    camera_matrix: Optional[npt.NDArray]
    dist_coeffs: Optional[npt.NDArray]
    R_c2g: Optional[npt.NDArray]
    hand_eye_translations: Optional[npt.NDArray]
#TODO
class Calibration():
	def __init__(self, config_path):
		self.__result: CalibrationResult = {
			'error': None,
			'camera_matrix': None,
			'dist_coeffs': None,
			'R_c2g': None,
			'hand_eye_translations': None
		}

	# 設定檔路徑
		self.__config_path = config_path
		self.__result_path = None
		self.__image_dir = None
		self.__timestamp = None

		# 影像數據
		self.__rgg_image
		self.__depth_image
		self.__robot_pose

		# 校正相關數據
		self.__object_points = []
		self.__object_corners = []
		self.__image_points = []
		self.__image_points_depth = []
		self.__rvecs_camera = []
		self.__tvecs_camera = []

		# 棋盤格參數
		self.__pattern_width = 0
		self.__pattern_height = 0
		self.__square_size = 0.0
		self.__trajectories = []

		self.__trajectory = None

		if not self.__read_config():
			raise ValueError('Config file not found')

		self.__set_timestamp()
		self.__initialize_objectCorners()
		self.__create_directories()

	def add_data(self, rgb, depth, robot_pose):




		pass

	def execute(self):

		if self.__validate_data():
			self.__calculate_camera_matrix()
			self.__perform_hand_eye_calibration()
			self.__calculate_chessboard_corners()
			self.__save_results()
		else:
			raise ValueError('Data is not valid')

		pass

	def get_trajectory(self):
		pass

	def get_result_path(self):
		pass

	def __initialize_objectCorners(self):
		pass

	def __set_timestamp(self):
		pass

	def __get_totation_from_Tm12(self):
		pass

	def __validate_data(self):
		pass

	def __read_config(self):
		pass

	def __create_directories(self):
		pass

	def __process_images(self):
		pass

	def __calculate_camera_matrix(self):
		pass

	def __perform_hand_eye_calibration(self):
		pass

	def __calculate_chessboard_corners(self):
		pass

	def __save_results(self):
		pass

class TM12_AMM_ROS2_Node(Node):
	def __init__(self, name='tm12_amm'):
		#TM12_AMM_zmq_clt()
		super().__init__(name)
		self.get_logger().info('TM12_AMM_ROS2_Node_py init')
		self.cv_bridge = CvBridge()

		self.get_logger().info("AiActionServerNode ready.")
		# Initialize timer

		self.set_parameter()
		self.set_callback_group()
		self.set_subscriber()
		self.set_publisher()
		self.set_service_client()
		self.set_service_server()
		self.set_action_client()
		self.set_action_server()

		self.timer = self.create_timer(1.0, self.timer_callback, callback_group=self.Reentrant_cb_group)

		self.parameter_callback_handle = self.add_on_set_parameters_callback(self.parameter_callback)

		self.action_map = {
			"Calibration": self.calibration,
			"Verify_Calibration": self.verify_calibration,
			"AI_Action": self.ai_action
		}

	def set_parameter(self):#ok
		read_only = ParameterDescriptor(read_only=True)

		# 宣告唯讀參數
		self.z_min_ = self.declare_parameter('z_min', 0.02, read_only).value
		self.velocity_max_ = self.declare_parameter('velocity_max', 2.0, read_only).value
		self.acc_time_min_ = self.declare_parameter('acc_time_min', 0.2, read_only).value
		self.home_in_joint_ = self.declare_parameter('home_in_joint', [-pi/4, 0.0, pi/2, 0.0, pi/2, 0.0], read_only).value

		# 相機參數宣告與轉換
		camera_matrix_list = self.declare_parameter('camera.matrix', [
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0
		], read_only).value
		self.camera_matrix_ = np.array(camera_matrix_list, dtype=np.float64).reshape(3, 3)
		#self.get_logger().info(f'Camera matrix:\n{str(self.camera_matrix_)}')

		dist_coeffs_list = self.declare_parameter('camera.distortion', [0.0, 0.0, 0.0, 0.0, 0.0], read_only).value
		self.dist_coeffs_ = np.array(dist_coeffs_list, dtype=np.float64)
		#self.get_logger().info(f'Distortion coefficients:\n{str(self.dist_coeffs_)}')


		'''
		# 計算新的相機矩陣用於校正
		image_size = (1280, 720)  # 根據實際相機解析度調整
		self.new_camera_matrix_, self.roi = cv2.getOptimalNewCameraMatrix(
			self.camera_matrix_,
			self.dist_coeffs_,
			image_size,
			1,
			image_size
		)

		# 預先計算重投影映射
		self.mapx_, self.mapy_ = cv2.initUndistortRectifyMap(
			self.camera_matrix_,
			self.dist_coeffs_,
			None,
			self.new_camera_matrix_,
			image_size,
			cv2.CV_32FC1
		)
		'''
		# 手眼校正參數宣告與轉換
		R_c2g_list = self.declare_parameter('hand_eye.rotation', [
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0
		], read_only).value
		self.R_c2g_ = np.array(R_c2g_list, dtype=np.float64).reshape(3, 3)

		t_c2g_list = self.declare_parameter('hand_eye.translation', [0.0, 0.0, 0.0], read_only).value
		self.t_c2g_ = np.array(t_c2g_list, dtype=np.float64)

		self.T_Cam2Grpr_ = np.eye(4)
		self.T_Cam2Grpr_[:3, :3] = self.R_c2g_
		self.T_Cam2Grpr_[:3, 3] = self.t_c2g_

		#self.get_logger().info(f'Hand-eye Tansform:\n{str(self.T_Cam2Grpr_)}')

		self.manual_mode_ = self.declare_parameter('manual_mode', False).value
		self.pose_take_photo_static_ = self.declare_parameter('pose_take_photo_static', [0.345, -0.518, 0.4, -pi, 0.0, pi/4]).value

		self.rgb_img_ = None
		self.depth_img_ = None
		self.tm12_feedback_ = None

	def set_callback_group(self):# ok
		self.MutuallyExclusive_cb_group = MutuallyExclusiveCallbackGroup()
		self.Reentrant_cb_group = ReentrantCallbackGroup()

	def set_subscriber(self):
		self.realsense_rgb_subscription = self.create_subscription(
			Image,
			"/camera/realsense_camera/color/image_raw",
			self.realsense_rgb_callback,
			3,
			callback_group=self.Reentrant_cb_group
		)
		self.realsense_depth_subscription = self.create_subscription(
			Image,
			"/camera/realsense_camera/aligned_depth_to_color/image_raw",
			self.realsense_depth_callback,
			3,
			callback_group=self.Reentrant_cb_group
		)
		# Initialize feedback subscription and service
		self.tm12_feedback_subscription = self.create_subscription(
			FeedbackState,
			"feedback_states",
			self.tm12_feedback_callback,
			10,
			callback_group=self.Reentrant_cb_group
		)
		# Initialize keyboard manual subscription and parameter callback
		self.keyboard_manual_subscription = self.create_subscription(
			Char,
			'keyboard_manual',
			self.keyboard_manual_callback,
			10)

	def set_publisher(self):#ok
		self.amr_twist_publisher = self.create_publisher(
			Twist,
			'amr_twist',
			10)

	def set_service_client(self):#ok
		# gripper 2f85 clients
		self.grpr2f85_set_gripper_state_client = self.create_client(
			SetGripperState,
			'set_gripper_state',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		self.grpr2f85_get_gripper_status_client = self.create_client(
			GetGripperStatus,
			'get_gripper_status',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		# iaMech clients
		self.amr_servo_on_client = self.create_client(
			Trigger,
			'amr_servo_on',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		self.amr_servo_off_client = self.create_client(
			Trigger,
			'amr_servo_off',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		# TM12 clients
		self.tm12_set_positions_client = self.create_client(
			SetPositions,
			'set_positions',
        		callback_group=self.MutuallyExclusive_cb_group
		)

	def set_service_server(self):
		# Initialize homing service
		self.homing = self.create_service(
			Trigger,
			'homing',
			self.homing_callback,
			callback_group=self.Reentrant_cb_group  # 加入回調群組
		)
		pass

	def set_action_client(self):#ok
		pass

	def set_action_server(self):
		self.ai_action_server = ActionServer(
			self,
			Dotask,
			'ai_action',
            self.execute_callback,
            callback_group=self.Reentrant_cb_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
			# feedback_callback
        )

	def parameter_callback(self, parameters):#ok
		"""
		處理參數更新的回調函數
		"""

		result = SetParametersResult(successful=True)

		try:
			for parameter in parameters:
				if parameter.name == 'manual_mode':
					# 更新手動模式狀態
					self.manual_mode_ = parameter.value
					self.get_logger().info(f'手動模式已{"啟用" if self.manual_mode_ else "停用"}')

				elif parameter.name == 'pose_take_photo':
					new_pose = np.array(parameter.value)
					if len(new_pose) != 6:
						raise ValueError("拍照位姿必須有 6 個元素")
					self.pose_take_photo_ = new_pose

		except Exception as e:
			self.get_logger().error(f'參數更新失敗: {str(e)}')
			result.successful = False
			result.reason = str(e)

		return result

####################################################

	def realsense_rgb_callback(self, msg: Image): #ok
		try:
			self.rgb_img_ = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
		except Exception as e:
			self.get_logger().error(f'Error in RGB image callback: {e}')

	def realsense_depth_callback(self, msg: Image): #ok
		try:
			self.depth_img_ = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
			self.depth_img_ /= 1000.0
		except Exception as e:
			self.get_logger().error(f'Error in depth image callback: {e}')
		pass

	def tm12_feedback_callback(self, msg: FeedbackState): #ok
		try:
			self.tm12_feedback_ = msg
		except Exception as e:
			self.get_logger().error(f'Error in TM12 feedback callback: {e}')

	def keyboard_manual_callback(self, msg: Char):
		if not self.manual_mode_:
			self.get_logger().warn('Manual mode is disabled. Enable it first.')
			return

		try:
			# Get the key
			self.process_key(chr(msg.data))
			# Process the key
		except Exception as e:
			self.get_logger().error(f'Error in keyboard manual callback: {e}')

	def process_key(self, key):#TODO
		# Process the key
		if key == 'q':
			self.get_logger().info('Move forward')
		elif key == 'a':
			self.get_logger().info('Move left')
		elif key == 'w':
			self.get_logger().info('Move backward')
		elif key == 's':
			self.get_logger().info('Move right')
		elif key == 'e':
			self.get_logger().info('Turn left')
		elif key == 'd':
			self.get_logger().info('Turn right')
		elif key == 'r':
			self.get_logger().info('Move up')
		elif key == 'f':
			self.get_logger().info('Move down')
		elif key == 't':
			self.get_logger().info('Open gripper')
		elif key == 'g':
			self.get_logger().info('Close gripper')
		elif key == 'y':
			self.get_logger().info('Take photo')
		elif key == 'h':
			self.get_logger().info('Move to home')
		elif key == 'z':
			self.get_logger().info('Toggle manual mode')
		elif key == 'x':
			self.get_logger().info('Exit manual mode')
		elif key == '27':
			self.get_logger().info('Exit manual mode')
		else:
			self.get_logger().info(f'Unknown key: {key}')


####################################################

	def timer_callback(self): #ok
		pass

####################################################

	def call_tm12_set_positions(self,
				   motion_type = 2,
				   positions = [0.3571, -0.5795, 0.5, -pi, 0.0, pi/4],
				   velocity = 1.5,
				   acc_time = 1.0,
				   blend_percentage = 0,
				   fine_goal = True
				   ):
		try:
			req = SetPositions.Request()
			if motion_type in [1, 2, 4]:
				req.motion_type = motion_type
			else:
				raise ValueError('Motion_type is not allowed: PTP_J = 1, PTP_T = 2, LINE_T = 4')
			req.positions = self.range_check(motion_type, positions)

			req.velocity = min(velocity, self.velocity_max_)
			req.acc_time = max(acc_time, self.acc_time_min_)
			req.blend_percentage = min(100, max(0, int(blend_percentage + 0.5)))
			req.fine_goal = fine_goal

			future = self.tm12_set_positions_client.call(req)
			return future

		except Exception as e:
			self.get_logger().error(f'設定 TM12 位置時發生錯誤: {str(e)}')
			raise

	def call_grpr2f85_set_gripper_state(self,
					   position = 0,
					   speed = 255,
					   force = 255,
					   wait_time = 0
					   ):
		try:
			req = SetGripperState.Request()
			req.position = min(255, max(0, int(position + 0.5)))
			req.speed = min(255, max(0, int(speed + 0.5)))
			req.force = min(255, max(0, int(force + 0.5)))
			req.wait_time = max(int(wait_time + 0.5), 0)

			future = self.grpr2f85_set_gripper_state_client.call(req)
			return future

		except Exception as e:
			self.get_logger().error(f'設定 2f85 夾爪狀態時發生錯誤: {str(e)}')
			raise

	def call_grpr2f85_get_gripper_status(self):
		try:
			req = GetGripperStatus.Request()

			future = self.grpr2f85_get_gripper_status_client.call_async(req)
			return future

		except Exception as e:
			self.get_logger().error(f'取得 2f85 夾爪狀態時發生錯誤: {str(e)}')
			raise

	def call_amr_servo_on(self):
		try:
			req = Trigger.Request()

			future = self.amr_servo_on_client.call_async(req)
			return future

		except Exception as e:
			self.get_logger().error(f'啟動 AMR 伺服時發生錯誤: {str(e)}')
			raise

	def call_amr_servo_off(self):
		try:
			req = Trigger.Request()

			future = self.amr_servo_off_client.call_async(req)
			return future

		except Exception as e:
			self.get_logger().error(f'關閉 AMR 伺服時發生錯誤: {str(e)}')
			raise

####################################################

	def homing_callback(self, request, response):
		"""處理歸零/回到原位的服務請求

		Args:
			request (Trigger.Request): 觸發請求
			response (Trigger.Response): 服務回應

		Returns:
			Trigger.Response: 包含執行結果的回應
		"""
		try:
			self.homing_execute()
			response.success = True
			response.message = "歸零動作完成"
		except Exception as e:
			self.get_logger().error(f'歸零過程發生錯誤: {e}')
			response.success = False
			response.message = f"歸零失敗: {str(e)}"
		return response

	def homing_execute(self):
		try:
			future_tm12 = self.call_tm12_set_positions(
			motion_type=1,
			positions=self.home_in_joint_
			)
			future_gripper = self.call_grpr2f85_set_gripper_state(
			position=0
			)

		except Exception as e:
			self.get_logger().error(f'歸零過程發生錯誤: {e}')
			raise

####################################################
#action_clt
####################################################

	def goal_callback(self, goal_request):
		"""處理新的 goal 請求"""
		self.get_logger().info('Received goal request')
		# 在這裡可以加入 goal 的驗證邏輯
		if goal_request.task in ['Calibration', 'Verify_Calibration', 'AI_Action']:
			return GoalResponse.ACCEPT
		return GoalResponse.REJECT

	def cancel_callback(self, goal_handle):
		"""處理取消請求"""
		self.get_logger().info('Received cancel request')
		return CancelResponse.ACCEPT

	def execute_callback(self, goal_handle: ServerGoalHandle):
		"""執行 goal 的回調函數"""
		if self.manual_mode_:
			self.get_logger().warn('Cannot execute automatic actions in manual mode')
			goal_handle.abort()
			return Dotask.Result()

		self.get_logger().info('Executing goal...')
		result = Dotask.Result()

		try:
			print("zzzzzzzzzzz")
			if goal_handle.request.task in self.action_map:
				self.action_map[goal_handle.request.task](
					goal_handle.request.scenario,
					goal_handle.request.repeat_times)
				result.ok = True
				result.result = 'Success'
			else:
				goal_handle.abort()
				result.ok = False
				result.result = 'Unknown task'
				return result

			goal_handle.succeed()
			self.get_logger().info('Goal succeeded')
			return result

		except Exception as e:
			self.get_logger().error(f'執行目標時發生錯誤: {str(e)}')
			goal_handle.abort()
			result.ok = False
			result.result = str(e)
			return result
#TODO
	def calibration(self, scenario, repeat_times=1):
		"""執行校正任務"""
		try:
			# TODO: 實作校正邏輯
			pass
		except Exception as e:
			self.get_logger().error(f'校正過程發生錯誤: {e}')
			raise

	def verify_calibration(self, scenario, repeat_times=1):
		"""執行校正驗證"""
		try:
			# TODO: 實作校正驗證邏輯
			pass
		except Exception as e:
			self.get_logger().error(f'校正驗證過程發生錯誤: {e}')
			raise

	def ai_action(self, scenario, repeat_times=1): #ok
		"""執行 AI 動作

		Args:
			scenario (str): 場景名稱
			repeat_times (int): 重複執行次數
		"""
		trashCan_position = [
			[0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0.],
		]
		try:
			while(True): 
				for i in range(repeat_times):
					# 先移動到拍照位置
					self.call_tm12_set_positions(
					positions=self.pose_take_photo_static_
					)
					self.wait_for_tm12_arrive(self.pose_take_photo_static_)
					time.sleep(0.5)

					# 取得當前機器人位置的變換矩陣
					T_Grpr2Base = self.get_transform_from_tm12_cartesian_pose(
					self.tm12_feedback_.tool_pose
					)

					# 透過自訂演算法獲取物體相對於相機的位姿
					det = detection(self)
					det_result = det.detection()
					if det_result == None:
						raise RuntimeError("偵測失敗: 未偵測到可抓取物件")
					
					x, y, z, class_id = det_result
					coor = [x, y, z + 0.03, 0., 0., 0.]
					#coor = [0., 0., 0.4, 0., 0., 0.]

					# 檢查所需資料是否都準備好
					#if any(x is None for x in [self.rgb_img_, self.depth_img_, self.camera_matrix_]):
					#	raise ValueError("缺少必要的影像或相機參數")

					# response 推薦在相機座標下

					# T_obj2cam 2 T_obj2base
					# T_Obj2Cam = self.get_transform_from_tm12_cartesian_pose([0., 0., 0.4, 0., 0., 0.])
					T_Obj2Cam = self.get_transform_from_tm12_cartesian_pose(coor)
					T_Obj2Base = self.get_transform_Obj2Base(T_Grpr2Base, T_Obj2Cam)
					pose_obj2base = self.get_tm12_cartesian_pose_from_matrix(T_Obj2Base)

					self.get_logger().info(f'物體位姿: \n{pose_obj2base}')
					#self.get_logger().info(f'T_Obj2Cam: \n{T_Obj2Cam}')
					#self.get_logger().info(f'T_Grpr2Base: \n{T_Grpr2Base}')
					#self.get_logger().info(f'T_Obj2Base: \n{T_Obj2Base}')

					#T_test = self.get_transform_from_tm12_cartesian_pose([0.3571, -0.5795, 0.2, -3.1415, 0., 0.7854])
					#self.get_logger().info(f'Test: \n{T_test}')
					#original
					ret = self.pick_at(pose_obj2base, 1)# [0.3571, -0.5795, 0.2, -3.1415, 0., 0.7854]
					if ret == False:
						self.get_logger().error('AI 動作抓取期間發生錯誤: 未成功抓取')
						continue
					#self.place_at(trashcan_position[class_id])#  [0.3571, -0.5795, 0.2, -3.1415, 0., 0.7854]
					self.place_at([0.1, -0.5, 0.1, -3.1415, 0., 0.7854])#  [0.3571, -0.5795, 0.2, -3.1415, 0., 0.7854]

				# 回到安全位置
				self.homing_execute()

		except Exception as e:
			self.get_logger().error(f'AI 動作執行期間發生錯誤: {e}')
			raise
		self.get_logger().info('成功執行Service: ai_action')
		return
	
########################################################

	def get_transform_from_tm12_cartesian_pose(self, tm12_pose=[0.3571, -0.5795, 0.5, -pi, 0.0, pi/4]):
		"""將 TM12 笛卡爾座標轉換為齊次變換矩陣

		Args:
			tm12_pose (list): [x, y, z, rx, ry, rz] 位置和歐拉角 (弧度)

		Returns:
			np.ndarray: 4x4 齊次變換矩陣
		"""
		try:
			# 檢查輸入
			if len(tm12_pose) != 6:
				raise ValueError("TM12 pose 必須包含 6 個元素 [x, y, z, rx, ry, rz]")

			# 從歐拉角獲取旋轉矩陣 (使用 XYZ 順序)
			rx, ry, rz = tm12_pose[3:6]

			# 分別計算各軸的旋轉矩陣
			Rx = np.array([
			[1, 0, 0],
			[0, np.cos(rx), -np.sin(rx)],
			[0, np.sin(rx), np.cos(rx)]
			])

			Ry = np.array([
			[np.cos(ry), 0, np.sin(ry)],
			[0, 1, 0],
			[-np.sin(ry), 0, np.cos(ry)]
			])

			Rz = np.array([
			[np.cos(rz), -np.sin(rz), 0],
			[np.sin(rz), np.cos(rz), 0],
			[0, 0, 1]
			])

			# 組合旋轉矩陣 (ZYX 順序)
			R = Rz @ Ry @ Rx

			# 建立齊次變換矩陣
			T = np.eye(4)
			T[:3, :3] = R
			T[:3, 3] = tm12_pose[:3]

			return T #T_Gripper2Base

		except Exception as e:
			self.get_logger().error(f'轉換 TM12 笛卡爾座標時發生錯誤: {str(e)}')
			raise

	def get_tm12_cartesian_pose_from_matrix(self, T_Gripper2Base):
		"""將齊次變換矩陣轉換為 TM12 笛卡爾座標

		Args:
			T_Gripper2Base (np.ndarray): 4x4 齊次變換矩陣

		Returns:
			list: [x, y, z, rx, ry, rz] 位置和歐拉角 (弧度)
		"""
		try:
			# 檢查輸入矩陣
			if not isinstance(T_Gripper2Base, np.ndarray) or T_Gripper2Base.shape != (4, 4):
				raise ValueError("輸入必須是 4x4 齊次變換矩陣")

			# 提取位置向量
			position = T_Gripper2Base[:3, 3]

			# 提取旋轉矩陣
			R = T_Gripper2Base[:3, :3]

			# 計算歐拉角 (使用 ZYX 順序)
			# ry = arcsin(-r31)
			ry = np.arcsin(-R[2, 0])

			# rx = arctan2(r32, r33)
			rx = np.arctan2(R[2, 1], R[2, 2])

			# rz = arctan2(r21, r11)
			rz = np.arctan2(R[1, 0], R[0, 0])

			# 組合結果
			tm12_pose = [
			position[0],  # x
			position[1],  # y
			position[2],  # z
			rx,          # 繞 X 軸旋轉
			ry,          # 繞 Y 軸旋轉
			rz           # 繞 Z 軸旋轉
			]

			return tm12_pose

		except Exception as e:
			self.get_logger().error(f'從齊次變換矩陣轉換 TM12 笛卡爾座標時發生錯誤: {str(e)}')
			raise

	def get_transform_Obj2Base(self, T_Grpr2Base, T_Obj2Cam):
		# Obj2Base = Grpr2Base Cam2Grpr Obj2Cam
		# Grpr2Base: 拍照時的位置
		# Cam2Grpr: 相機到夾爪的轉換(常數)
		# Obj2Cam: 物件到相機的轉換(從模型得到)
		return T_Grpr2Base @ (self.T_Cam2Grpr_ @ T_Obj2Cam)

	def range_check(self, motion_type, position):
		if motion_type == 2:
			position[2] = max(position[2], self.z_min_)
		return position

	def wait_for_tm12_arrive(self, target_pose, tolerance=0.001):
		"""檢查 TM12 是否到達目標位置

		Args:
			target_pose (list): 目標位置 [x, y, z, rx, ry, rz]
			tolerance (float, optional): 容許誤差. Defaults to 0.01.

		Returns:
			bool: 是否到達目標位置
		"""
		try:
			# 檢查目標位置是否為有效列表
			if len(target_pose) != 6:
				raise ValueError("目標位置必須包含 6 個元素 [x, y, z, rx, ry, rz]")

			# 檢查 TM12 回饋是否有效
			if self.tm12_feedback_ is None:
				raise ValueError("TM12 回饋為空")

			# 計算位置誤差
			while np.linalg.norm(np.array(target_pose) - np.array(self.tm12_feedback_.tool_pose)) > tolerance:
				#self.get_logger().info('等待機器人到達目標位置')
				time.sleep(0.01)


		except Exception as e:
			self.get_logger().error(f'檢查 TM12 是否到達目標位置時發生錯誤: {str(e)}')

	def pick_at(self, pose_obj2base, openning):
		self.get_logger().info('測試')
		PRE_GRASP_OFFSET = 0.1
		try:
			# 計算物體相對於基座的變換
			T_Obj2Base = self.get_transform_from_tm12_cartesian_pose(pose_obj2base)

			# 計算預抓取位置
			approach_vector = T_Obj2Base[:3, 2]
			T_PreGrasp2Base = T_Obj2Base.copy()
			T_PreGrasp2Base[:3, 3] -= approach_vector * PRE_GRASP_OFFSET

			# 轉換為笛卡爾座標
			pre_grasp_pose = self.get_tm12_cartesian_pose_from_matrix(T_PreGrasp2Base)
			target_pose = self.get_tm12_cartesian_pose_from_matrix(T_Obj2Base)

			# 1. 開啟夾爪並確認
			self.call_grpr2f85_set_gripper_state(position=0)
			
			# 2. 移動到預抓取位置
			self.call_tm12_set_positions(positions=pre_grasp_pose)

			# 3. 移動到目標位置
			self.call_tm12_set_positions(positions=target_pose)

			self.wait_for_tm12_arrive(target_pose)

			self.get_logger().info('機器人已到達目標位置')
			grpr_result = self.call_grpr2f85_set_gripper_state(position=int(openning * 255 + 0.5), wait_time=0)
			if grpr_result.status_code == 2: # 成功抓取
				self.get_logger().info('成功抓取')
				# 5. 提起物體回到預抓取位置
				self.call_tm12_set_positions(positions=pre_grasp_pose)
			elif grpr_result.status_code == 3:  # 抓取失敗
				self.get_logger().error('未抓取到物體')
				try:
					self.homing_execute()
				except Exception as e:
					self.get_logger().error(f'回原位失敗: {e}')
				return False

			return True

		except Exception as e:
			self.get_logger().error(f'抓取過程發生錯誤: {str(e)}')
			# 發生錯誤時嘗試回到安全位置
			try:
				self.homing_execute()
			except:
				pass
			raise

	def place_at(self, pose_obj2base):
		PRE_GRASP_OFFSET = 0.1
		try:
			# 計算物體相對於基座的變換
			T_Obj2Base = self.get_transform_from_tm12_cartesian_pose(pose_obj2base)

			# 計算預抓取位置
			approach_vector = T_Obj2Base[:3, 2]
			T_PreGrasp2Base = T_Obj2Base.copy()
			T_PreGrasp2Base[:3, 3] -= approach_vector * PRE_GRASP_OFFSET

			# 轉換為笛卡爾座標
			pre_grasp_pose = self.get_tm12_cartesian_pose_from_matrix(T_PreGrasp2Base)
			target_pose = self.get_tm12_cartesian_pose_from_matrix(T_Obj2Base)

			# 2. 移動到預放置位置
			self.call_tm12_set_positions(positions=pre_grasp_pose)

			# 3. 移動到目標位置
			self.call_tm12_set_positions(positions=target_pose)

			self.wait_for_tm12_arrive(target_pose)

			self.call_grpr2f85_set_gripper_state(position=0, wait_time=0)

			# 5. 回到預放置位置
			self.call_tm12_set_positions(positions=pre_grasp_pose)

			return True

		except Exception as e:
			self.get_logger().error(f'抓取過程發生錯誤: {str(e)}')
			# 發生錯誤時嘗試回到安全位置
			try:
				self.homing_execute()
			except:
				pass
			raise


########################################################
# detection
########################################################
class detection():
	CLASS_NAMES = [
	"Plastic Bottle",
	"Can",
	"tissue",
	"carton"
	]

	def __init__(self, node):
		"""
		初始化detection所需的參數
		"""
		self.node = node
		self.logger = node.get_logger()
		self.logger.info("Detection 初始化完成")

		# 1) 初始化 YOLO 與 RealSense (原 kalmanTest.py 裏面 main() 的前半段)
		self.pipeline, self.align, self.model = self.init_realsense_yolo()
		self.logger.info("RealSense pipeline & YOLO model initialized.")

		# 2) 建立 OpenCV 顯示視窗 (若您在無頭環境則可忽略，或自行關閉)
		cv2.namedWindow("RealSense YOLO Detection", cv2.WINDOW_NORMAL)

		# 3) 建立追蹤器相關參數 (依照原程式)
		self.trackers = []           # 用來裝 KalmanFilterBBox 物件
		self.dt = 1/30               # frame rate
		self.MAX_LOST = 10
		self.IOU_THRESHOLD = 0.3

		# 4) 初始化一些控制變數
		self.last_distance = None
		# self.start_time = time.time()

		# 5) 先讀取深度內參
		frames = self.pipeline.wait_for_frames()
		aligned_frames = self.align.process(frames)
		depth_frame = aligned_frames.get_depth_frame()
		if depth_frame:
			self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
			self.logger.info("Depth intrinsics captured.")
		else:
			self.depth_intrinsics = None
			self.logger.warn("No depth frame available at init; intrinsics set to None.")

		# 完成初始化
		self.logger.info("Detection __init__() complete. Start detection loop.")

	def init_realsense_yolo(self):
		"""
		原本 kalmanTest.py 的 init_realsense_yolo 函式。
		回傳 pipeline, align, model
		"""
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
		#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
		pipeline.start(config)

		align = rs.align(rs.stream.color)
		model = YOLO("/home/robotics/garbage_classification/runs/detect/train7/weights/best.pt") 
		return pipeline, align, model

	# ========== 獲取RealSense影像與處理 ==========
	def get_realsense_frames(self, pipeline, align):
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)
		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()
		if not color_frame or not depth_frame:
			return None, None, None

		return color_frame, depth_frame

	# ========== (A) 簡易的 2D Bounding Box 卡曼濾波器 ==========
	class KalmanFilterBBox:
		def __init__(self, dt=1.0, class_id=None):
			# 狀態向量: [x, y, w, h, vx, vy, vw, vh]
			self.dt = dt
			self.dim_state = 8

			# 追蹤器對應的類別，直接存 class_id，或可存 class_name
			self.class_id = class_id

			# 狀態轉移矩陣 F (8x8)
			self.F = np.array([
				[1, 0, 0, 0, dt, 0,  0,  0],
				[0, 1, 0, 0, 0,  dt, 0,  0],
				[0, 0, 1, 0, 0,  0,  dt, 0],
				[0, 0, 0, 1, 0,  0,  0,  dt],
				[0, 0, 0, 0, 1,  0,  0,  0],
				[0, 0, 0, 0, 0,  1,  0,  0],
				[0, 0, 0, 0, 0,  0,  1,  0],
				[0, 0, 0, 0, 0,  0,  0,  1]
			], dtype=np.float32)

			# 量測矩陣 H (4x8)，只量測 [x, y, w, h]
			self.H = np.array([
				[1, 0, 0, 0, 0, 0, 0, 0],
				[0, 1, 0, 0, 0, 0, 0, 0],
				[0, 0, 1, 0, 0, 0, 0, 0],
				[0, 0, 0, 1, 0, 0, 0, 0]
			], dtype=np.float32)

			# 狀態向量 X (8x1)
			self.X = np.zeros((8,1), dtype=np.float32)

			# 預估誤差協方差 P (8x8)
			self.P = np.eye(8, dtype=np.float32)

			# 過程雜訊協方差 Q
			self.Q = np.eye(8, dtype=np.float32) * 0.01

			# 量測雜訊協方差 R (4x4)
			self.R = np.eye(4, dtype=np.float32) * 0.1

			# 追蹤品質計數器 (可用來判斷追蹤器是否可信，或失效等)
			self.lost_frames = 0

			# 保存最近 10 幀的 3D 座標紀錄 
			self.history = []

			# 原先只存 (X, Y, Z)，現在加入 confidence，
			# 故改用 (X, Y, Z, conf) 四元組。
			self.current_conf = 0.0  # 用於記錄本幀偵測信心

		def init_state(self, bbox):
			# bbox: (x, y, w, h)
			self.X[0] = bbox[0]
			self.X[1] = bbox[1]
			self.X[2] = bbox[2]
			self.X[3] = bbox[3]
			# 速度初始化暫設 0
			self.X[4] = 0
			self.X[5] = 0
			self.X[6] = 0
			self.X[7] = 0

		def predict(self):
			self.X = self.F @ self.X
			self.P = self.F @ self.P @ self.F.T + self.Q
			return self.X

		def update(self, meas):
			# meas: (x, y, w, h)
			z = np.array(meas, dtype=np.float32).reshape((4,1))

			S = self.H @ self.P @ self.H.T + self.R
			K = self.P @ self.H.T @ np.linalg.inv(S)

			y = z - (self.H @ self.X)
			self.X = self.X + K @ y

			I = np.eye(self.dim_state, dtype=np.float32)
			self.P = (I - K @ self.H) @ self.P

			# 每次更新到量測，可視為追蹤成功 => 重置 lost_frames
			self.lost_frames = 0

			return self.X

		def get_bbox(self):
			# 取出 [x, y, w, h] 來繪製框，並確保不為負
			x, y, w, h = self.X[0], self.X[1], self.X[2], self.X[3]
			w = max(w, 1)
			h = max(h, 1)
			return (int(x), int(y), int(w), int(h))
		
		def add_3d_history(self, xyz, conf):
			"""
			xyz: (X, Y, Z) 3D 座標
			conf: float, 本次追蹤/偵測的信心度
			僅保留最近 10 筆歷史
			"""
			self.history.append((xyz[0], xyz[1], xyz[2], conf))
			if len(self.history) > 10:
				self.history.pop(0)

	# ========== (B) 簡易 IOU 函數，做關聯用 ==========
	def iou(self, bbox1, bbox2):
		# bbox: (x, y, w, h)
		x1, y1, w1, h1 = bbox1
		x2, y2, w2, h2 = bbox2

		# 轉為 x1y1x2y2，(x1, y1)為左上角、(x2, y2)為右下角
		box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
		box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2

		#交集區域的左上角座標，對應於「兩個框的左上角中 x、y 最大的那個」
		inter_x1 = max(box1_x1, box2_x1)
		inter_y1 = max(box1_y1, box2_y1)
		#交集區域的右下角座標，對應於「兩個框的右下角中 x、y 最小的那個」
		inter_x2 = min(box1_x2, box2_x2)
		inter_y2 = min(box1_y2, box2_y2)

		inter_w = max(0, inter_x2 - inter_x1)
		inter_h = max(0, inter_y2 - inter_y1)
		inter_area = inter_w * inter_h

		area1 = w1 * h1
		area2 = w2 * h2
		union_area = area1 + area2 - inter_area

		if union_area <= 0:
			return 0
		return inter_area / union_area

	# ========== (c) 3D座標平均函數 ==========
	def average_3d_coordinates(self, history):
		if not history:  # 若 history 為空
			return None  

		history_array = np.array(history)  # 轉為 NumPy 陣列，形狀為 (N, 4)
		mean_xyz = np.mean(history_array, axis=0)  # 計算每個維度的平均
		return tuple(mean_xyz)  # 轉回 tuple 較易讀

	# ==========  主程式: 偵測邏輯 ===========
	def detection(self): 
		"""
		主要偵測函數
		"""
		self.start_time = time.time()
		while(True): 
			# 1) 時間判斷：若超過一定秒數，或長時間都偵測不到目標，就結束節點
			elapsed_time = time.time() - self.start_time
			# 若超過3秒仍沒追到任何東西，可視需求直接結束
			if elapsed_time > 7 and len(self.trackers) == 0:
				self.logger.info("未穩定偵測到任何目標，程式結束。")
				cv2.destroyAllWindows()
				self.shutdown()
				return None
			# 若超過10秒就自動結束
			if elapsed_time > 7:
				self.logger.info("偵測超過5秒，程式結束。")
				self.shutdown()
				try: 
					x, y, z, class_id = self.final_results()
				except Exception as e:
					self.logger.error(f'計算最終目標期間發生錯誤: {e}')
				return (x, y, z, class_id)

			# 2) 取得對齊之 color 與 depth frame
			# depth_intrinsics 中就包含 fx, fy, ppx, ppy, distortion 等資訊
			color_frame, depth_frame = self.get_realsense_frames(self.pipeline, self.align)
			if not color_frame or not depth_frame:
				return

			frame = np.asanyarray(color_frame.get_data())

			# 3) YOLO 推論
			results = self.model(frame)
			det = results[0]
			boxes = det.boxes  # YOLO 會將預測結果放在 det.boxes

			# 4) 先對現有追蹤器做 predict()
			for kf in self.trackers:
				kf.predict()  # 卡曼預測
				# 若本迴圈沒匹配到量測，後續會 lost_frames += 1

			# 5) 取得新的偵測框 (det_bboxes) 與對應的類別/信心
			det_bboxes = []
			class_ids = []
			confs = []

			for box in boxes:
				x1, y1, x2, y2 = box.xyxy[0]
				w = x2 - x1
				h = y2 - y1
				x = x1
				y = y1
				det_bboxes.append((int(x), int(y), int(w), int(h)))

				cls_id = int(box.cls[0])
				class_ids.append(cls_id)

				conf = float(box.conf[0])
				confs.append(conf)

			# 6) 進行追蹤資料關聯：IOU 配對
			matched_trackers = set()
			for i, dbbox in enumerate(det_bboxes):
				best_iou = 0.0
				best_tracker_idx = None
				for kf_idx, kf in enumerate(self.trackers):
					pred_bbox = kf.get_bbox()
					current_iou = self.iou(pred_bbox, dbbox)
					if current_iou > best_iou:
						best_iou = current_iou
						best_tracker_idx = kf_idx
				
				# 若大於 IOU 閾值 => 更新該 tracker
				if best_iou > self.IOU_THRESHOLD and best_tracker_idx is not None:
					self.trackers[best_tracker_idx].update(dbbox)
					self.trackers[best_tracker_idx].current_conf = confs[i]
					matched_trackers.add(best_tracker_idx)
				else:
					# 找不到合適 => 新增 tracker
					cls_id = class_ids[i]
					new_kf = self.KalmanFilterBBox(self.dt, class_id=cls_id)
					new_kf.init_state(dbbox)
					new_kf.current_conf = confs[i]
					self.trackers.append(new_kf)

			# 7) 沒被配對到的 tracker => lost_frames + 1
			for idx, kf in enumerate(self.trackers):
				if idx not in matched_trackers:
					kf.lost_frames += 1

			# 8) 移除失效追蹤器
			self.trackers = [kf for kf in self.trackers if kf.lost_frames <= self.MAX_LOST]

			# 9) 繪製結果
			annotated_frame = frame.copy()
			for idx, kf in enumerate(self.trackers):
				x, y, w, h = kf.get_bbox()
				x2, y2 = x + w, y + h

				# 畫 2D 方框
				cv2.rectangle(annotated_frame, (x, y), (x2, y2), (0, 255, 0), 2)

				# 顯示類別名稱
				if kf.class_id is not None and 0 <= kf.class_id < len(self.CLASS_NAMES):
					class_name = self.CLASS_NAMES[kf.class_id]
					cv2.putText(annotated_frame, class_name, (x + 2, y + 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

				# 取得該 tracker 中心像素
				center_x = x + w // 2
				center_y = y + h // 2


				# 取得深度
				distance = depth_frame.get_distance(center_x, center_y)
				if self.last_distance is None:
					self.last_distance = distance
				# 若本次距離 0 或跳動過大，可酌情平滑處理
				elif distance <= 0 or (abs(distance - self.last_distance) > 0.05 and self.last_distance != 0):
					distance = self.last_distance
				self.last_distance = distance

				center_y = 720 - center_y
				center_x = 1280 - center_x
				# 反投影到 3D 座標
				point_3d = rs.rs2_deproject_pixel_to_point(
					self.depth_intrinsics, [center_x, center_y], distance
				)
				X, Y, Z = point_3d  # 單位：公尺
				#point_3d_arr = np.array(point_3d).reshape(3, 1)
				
				# 手眼座標轉換
				#point_arm = self.handEyeRotation @ point_3d_arr + self.handEyeTranslation
				#X, Y, Z = point_arm.ravel()

				# 在畫面上顯示 tracker ID 與 3D 座標
				text_id = f"ID={idx}"
				cv2.putText(annotated_frame, text_id, (x, y - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
				
				text_3d = f"3D=({X:.3f}, {Y:.3f}, {Z:.3f})"
				cv2.putText(annotated_frame, text_3d, (x, y + 15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

				# 記錄 3D 歷史
				kf.add_3d_history((X, Y, Z), kf.current_conf)

				# 顯示 confidence
				text_conf = f"Conf={kf.current_conf:.2f}"
				cv2.putText(annotated_frame, text_conf, (x + 2, y + 50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

			cv2.imshow("RealSense YOLO Detection", annotated_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				raise RuntimeError("User interrupt")

	# ==========  決定最終夾取物件 ===========
	def final_results(self): 
			max_conf = 0
			target_index = None

			# 過濾掉 history 少於 10 筆的 tracker
			self.trackers = [kf for kf in self.trackers if len(kf.history) >= 10]

			for i, kf in enumerate(self.trackers):
				print(f"Tracker ID = {i}, class_ = {self.model.names[kf.class_id]}, history(len = {len(kf.history)}) =")

				for idx, (X, Y, Z, conf) in enumerate(kf.history):
					print(f"  Frame {idx} coordinate: ({X:.3f}, {Y:.3f}, {Z:.3f}), confidence: {conf:.2f}")

				(AVG_X, AVG_Y, AVG_Z, AVG_CONF) = self.average_3d_coordinates(kf.history)
				if AVG_X is not None:
					print(f"  平均座標: ({AVG_X:.3f}, {AVG_Y:.3f}, {AVG_Z:.3f}), 平均信心: {AVG_CONF:.2f}")
				else: 
					print("平均座標: None")

				if AVG_CONF is not None and AVG_CONF > max_conf: 
					max_conf = AVG_CONF
					target_index = i

			if target_index is not None and max_conf != 0:
				(TAR_X, TAR_Y, TAR_Z, TAR_CONF) = self.average_3d_coordinates(self.trackers[target_index].history)
				class_id = self.CLASS_NAMES[self.trackers[target_index].class_id]
				print(f"目標id: {target_index}, 類別: {class_id}, 平均信心: {TAR_CONF:.2f}")
				print(f"目標座標: ({TAR_X:.3f}, {TAR_Y:.3f}, {TAR_Z:.3f})")
				return TAR_X, TAR_Y, TAR_Z, class_id
			else:
				raise Exception("沒有最終目標")

	def shutdown(self):
			"""
			關閉 pipeline，釋放相機資源
			"""
			try:
				#cv2.destroyAllWindows()
				self.pipeline.stop()
				self.logger.info("RealSense pipeline 已關閉")
			except Exception as e:
				self.logger.warn(f"關閉 pipeline 發生錯誤: {e}")
def main(args=None):
	rclpy.init(args=args)
	executor = MultiThreadedExecutor()

	node = TM12_AMM_ROS2_Node()
	executor.add_node(node)

	try:
		executor.spin()
	except KeyboardInterrupt:
		pass
	except Exception as exception:
		raise exception
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()

