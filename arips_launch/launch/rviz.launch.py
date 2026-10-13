import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

# ---- adjust these to your setup -------------------------------------------
ROBOT_NAME = "arips"                       # must match <robot name=...> in URDF and SRDF
MOVEIT_CONFIG_PACKAGE = "arips_moveit_config"
DESCRIPTION_PACKAGE = "arips_description"
URDF_RELATIVE_PATH = os.path.join("urdf", "arips_onshape.urdf.xacro")
# xacro arguments: must be identical to the ones used in your bringup (values as strings)
XACRO_MAPPINGS = {
    # "use_fake_hardware": "false",
}
# ----------------------------------------------------------------------------


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory(DESCRIPTION_PACKAGE), URDF_RELATIVE_PATH
    )

    moveit_config = (
        MoveItConfigsBuilder(ROBOT_NAME, package_name=MOVEIT_CONFIG_PACKAGE)
        .robot_description(file_path=urdf_path, mappings=XACRO_MAPPINGS)
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

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=str(os.path.join(get_package_share_directory("arips_launch"), "config", "rviz_config.rviz")),
        description="Path to the RViz config file",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation clock"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            moveit_config.pilz_cartesian_limits,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription([rviz_config_arg, use_sim_time_arg, rviz_node])