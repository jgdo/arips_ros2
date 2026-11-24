from setuptools import setup

package_name = 'arips_keypoint_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.learning'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
    description='Generic keypoint detection framework for RGB-D images.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'extract_rgbd_dataset = arips_keypoint_detection.bag_to_dataset:main',
            'keypoint_annotator = arips_keypoint_detection.annotation_tool:main',
            'create_split = arips_keypoint_detection.create_split:main',
            'keypoint_detector_node = arips_keypoint_detection.keypoint_detector_node:main',
            'train_keypoints = arips_keypoint_detection.learning.train:main',
            'evaluate_keypoints = arips_keypoint_detection.learning.evaluate:main',
            'export_keypoints_onnx = arips_keypoint_detection.learning.export_onnx:main',
            'predict_annotations = arips_keypoint_detection.learning.predict_annotations:main',
        ],
    },
)
