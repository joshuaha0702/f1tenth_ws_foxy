import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'f1tenth_lattice_ros2'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    # launch 폴더 안의 모든 .py 및 .launch.py 파일 중복 없이 포함
    (os.path.join('share', package_name, 'launch'), list(set(glob('launch/*.py') + glob('launch/*.launch.py')))),
    # config 폴더 안의 모든 .yaml 파일 포함
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]

# maps 폴더 및 모든 하위 디렉터리 파일들 동적 추가 (존재하는 파일만 엄격히 검사)
if os.path.exists('maps'):
    for root, dirs, files in os.walk('maps'):
        valid_files = [os.path.join(root, f) for f in files if os.path.isfile(os.path.join(root, f))]
        if valid_files:
            data_files.append((os.path.join('share', package_name, root), valid_files))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
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
            'slam_planner_node = f1tenth_lattice_ros2.slam_planner_node:main',
            'scan_odom_localization_node = f1tenth_lattice_ros2.scan_odom_localization_node:main',
            'episode_manager_node = f1tenth_lattice_ros2.episode_manager_node:main',
            'bag2gazebo_node = f1tenth_lattice_ros2.bag2gazebo_node:main',
        ],
    },
)