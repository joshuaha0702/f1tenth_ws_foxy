from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'f1tenth_lattice_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='F1Tenth Lattice Planner with Pure Pursuit for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lattice_planner_node = f1tenth_lattice_ros2.planner_node:main',
            'slam_lattice_planner_node = f1tenth_lattice_ros2.slam_planner_node:main',
        ],
    },
)
