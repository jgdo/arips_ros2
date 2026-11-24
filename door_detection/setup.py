from glob import glob

from setuptools import setup


package_name = 'door_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('pytorch/models/*')),
        ('share/' + package_name + '/test_data', glob('test_data/*')),
    ],
    install_requires=['setuptools', 'numpy'],
    extras_require={
        'test': [
            'pytest',
        ],
    },
    zip_safe=True,
    maintainer='jgdo',
    maintainer_email='jgdo@todo.todo',
    description='Door handle and floor step detection for ARIPS.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'door_handle_detector_node = '
            'door_detection.door_handle_detector_node:main',
            'step_detector_node = '
            'door_detection.step_detector_node:main',
        ],
    },
)