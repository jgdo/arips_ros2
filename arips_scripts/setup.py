from setuptools import setup


package_name = 'arips_scripts'

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
    description='One-shot ROS 2 scripts for ARIPS robot behaviors.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'call_cross_door_step = arips_scripts.call_cross_door_step:main',
        ],
    },
)
