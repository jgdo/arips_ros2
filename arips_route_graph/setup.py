from glob import glob

from setuptools import setup


package_name = 'arips_route_graph'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Mark Prediger',
    maintainer_email='markprediger@gmail.com',
    description='Generate connected free-space regions from ARIPS maps.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'route_graph_generator = '
            'arips_route_graph.route_graph_generator:main',
        ],
    },
)
