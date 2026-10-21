from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from launch_param_builder import load_yaml
import os
from pathlib import Path


def generate_launch_description():
    description_share = get_package_share_directory("arips_description")
    urdf_path = os.path.join(description_share, "urdf", "arips_onshape.urdf.xacro")
        
    moveit_config = (
        MoveItConfigsBuilder("arips", package_name="arips_moveit_config")
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(file_path="config/arips.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl",
        )
        .to_moveit_configs()
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": False,
             "publish_robot_description_semantic": True},
        ],
    )
    
    # servo_params = load_yaml(
    #     Path(get_package_share_directory("arips_moveit_config")) / "config" / "servo.yaml"
    # )
    # servo_node = Node(
    #     package="moveit_servo",
    #     executable="servo_node",
    #     name="servo_node",
    #     output="screen",
    #     parameters=[
    #         servo_params,
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic,
    #         moveit_config.robot_description_kinematics,
    #         moveit_config.joint_limits,
    #     ],
    # )
    
    return LaunchDescription([move_group])