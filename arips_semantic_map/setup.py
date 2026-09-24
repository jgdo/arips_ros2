from glob import glob
from setuptools import setup

package_name = 'arips_semantic_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/maps', glob('maps/*')),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Mark Prediger',
    maintainer_email='markprediger@gmail.com',
    description='ROS 2 semantic map server for ARIPS',
    license='BSD',
    entry_points={
        'console_scripts': [
            'semantic_map_server = arips_semantic_map.semantic_map_server:main',
        ],
    },
)
