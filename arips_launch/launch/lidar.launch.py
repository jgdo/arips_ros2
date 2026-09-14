#!/usr/bin/python3
# Copyright 2020, EAIBOT
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.actions import LogInfo

import lifecycle_msgs.msg
import os


def generate_launch_description():
    share_dir = get_package_share_directory('arips_launch')
    parameter_file = LaunchConfiguration('params_file')

    params_declare = DeclareLaunchArgument('params_file',
                                           default_value=os.path.join(
                                               share_dir, 'params', 'X4.yaml'),
                                           description='Path to the ROS2 parameters file to use.')

    # Updated for ROS2 Jazzy - use 'executable' instead of 'node_executable'
    # and 'name' instead of 'node_name', 'namespace' instead of 'node_namespace'
    driver_node = LifecycleNode(package='ydlidar_ros2_driver',
                                executable='ydlidar_ros2_driver_node',
                                name='ydlidar_ros2_driver_node',
                                output='screen',
                                emulate_tty=True,
                                parameters=[parameter_file],
                                namespace='/',
                                )
    
    tf2_node = Node(package='tf2_ros',
                    executable='static_transform_publisher',
                    name='static_tf_pub_laser',
                    arguments=['--x', '0', '--y', '0', '--z', '0.165',
                               '--roll', '0', '--pitch', '0', '--yaw', '3.14159',
                               '--frame-id', 'base_footprint', '--child-frame-id', 'laser_frame'],
                    )

    csm_node = Node(
                package='ros2_laser_scan_matcher',
                executable='laser_scan_matcher',
                output='screen',
                parameters=[{
                    'publish_odom': '/csm_odom',
                    'publish_tf': True,
                    'base_frame': 'arips_wheel_center',
                    'odom_frame': 'odom',
                    'laser_frame': 'laser_frame'
                }],
            )

    enable_publish_tf = ExecuteProcess(
           cmd=['ros2', 'topic', 'pub', '--times', '3',
               '/enable_publish_tf', 'std_msgs/msg/Bool', 'data: false'],
        output='screen',
    )

    return LaunchDescription([
        params_declare,
        enable_publish_tf,
        driver_node,
        tf2_node,
        csm_node,
    ])