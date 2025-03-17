#!/usr/bin/env python3

import sys
import os
import yaml
import launch
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    camera_config = os.path.join(
        get_package_share_directory('tm12_amm'),  # 你的 ROS2 package 名稱
        'config',
        'camera_config.yaml'
    )

    return LaunchDescription([
        Node(
            package='tm_driver',
            executable='tm_driver',
            output='screen',
            arguments=['robot_ip:=192.168.10.2'],
        ),

        Node(
            package='grpr2f85_driver',
            executable='grpr2f85_driver.py',
            output='screen',
            arguments=['usb_port:=1'],
        ),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_camera',
            parameters=[{
                'align_depth.enable': True,
                'pointcloud.enable': True,
                'spatial_filter.enable': True,
                #'disparity_filter.enable': True,
                'rgb_camera.color_profile': '1280x720x30',    # RGB 影像解析度
                'depth_module.depth_profile': '1280x720x30',  # 深度影像解析度
                'rgb_camera.enable_auto_exposure': True,
                'depth_module.enable_auto_exposure': True
            }]
        ),

        Node(
            package='iamech_driver',
            executable='iamech_driver.py',
            name='iamech_driver',
            output='screen',
            #arguments=[''],
        ),

        Node(
            package='tm12_amm',
            executable='tm12_amm.py',
            name='tm12_amm',
            output='screen',
            parameters=[camera_config]
            #arguments=['--ros-args', '--log-level', 'debug']
        ),

        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='rqt',
            output='screen',
            #arguments=['-d', rviz_config]
        ),

        #Node(
        #    package='rviz2',
        #    executable='rviz2',
        #    name='rviz2',
        #    output='screen',
        #    #arguments=['-d', rviz_config]
        #)
    ])