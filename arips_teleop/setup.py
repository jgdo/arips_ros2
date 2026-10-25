import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arips_teleop'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Joystick teleoperation publishing joint commands',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'arm_teleop_node = arips_teleop.arm_teleop_node:main',
            'record_control = arips_teleop.record_control:main',
        ],
    },
)
