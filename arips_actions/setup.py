from setuptools import setup


package_name = 'arips_actions'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/arips_actions.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mark Prediger',
    maintainer_email='markprediger@gmail.com',
    description='ROS 2 action servers for ARIPS robot behaviors.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'cross_door_step_server = '
            'arips_actions.cross_door_step_server:main',
            'grab_door_handle_server = '
            'arips_actions.grab_door_handle_server:main',
            'grab_door_handle_direct_sim_server = '
            'arips_actions.grab_door_handle_direct_sim_server:main',
            'grab_door_handle_policy_server = '
            'arips_actions.grab_door_handle_policy_server:main',
            'drive_around_pivot_server = '
            'arips_actions.drive_around_pivot_server:main',
            'drive_forward_server = '
            'arips_actions.drive_forward_server:main',
            'retreat_from_pivot_server = '
            'arips_actions.retreat_from_pivot_server:main',
            'move_arm_server = '
            'arips_actions.move_arm_server:main',
            'get_door_grasping_approach_pose = '
            'arips_actions.get_door_grasping_approach_pose:main',
            'door_state_detector = '
            'arips_actions.door_state_detector:main',
        ],
    },
)