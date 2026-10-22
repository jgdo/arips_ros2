import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory('arips_description'), 'urdf', 'arips_onshape.urdf.xacro')

    moveit_config = (
        MoveItConfigsBuilder('arips', package_name='arips_moveit_config')
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(file_path='config/arips.srdf')
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .planning_pipelines(pipelines=['ompl'], default_planning_pipeline='ompl')
        .to_moveit_configs()
    )
    # moveit_cpp expects the pipelines list as planning_pipelines.pipeline_names
    moveit_params = moveit_config.to_dict()
    moveit_params['planning_pipelines'] = {'pipeline_names': ['ompl']}
    params_file = os.path.join(
        get_package_share_directory('arips_teleop'), 'config', 'teleop.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('group_name', default_value='arm'),
        Node(
            package='arips_teleop',
            executable='teleop_node',
            output='screen',
            parameters=[
                moveit_params,
                params_file,
                {'group_name': LaunchConfiguration('group_name')},
            ],
        ),
    ])
