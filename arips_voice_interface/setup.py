from setuptools import setup


package_name = 'arips_voice_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models',
            ['models/de_DE-thorsten-medium.onnx',
             'models/de_DE-thorsten-medium.onnx.json',
             'models/en_US-joe-medium.onnx',
             'models/en_US-joe-medium.onnx.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mark Prediger',
    maintainer_email='markprediger@gmail.com',
    description='Voice interface (text-to-speech) for ARIPS.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'arips_say = arips_voice_interface.arips_say:main',
        ],
    },
)
