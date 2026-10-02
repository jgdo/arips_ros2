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
        ],
    },
)