# Copyright 2019 Canonical, Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 질문자님의 패키지 안의 YAML 파일 경로를 정확히 지정합니다.
    pkg_share = get_package_share_directory('f1tenth_fgm_ros2')
    parameters_file = os.path.join(pkg_share, 'config', 'joy_teleop.yaml')

    # 2. 실제 조이스틱 하드웨어를 읽는 노드 (이게 있어야 스틱 신호가 들어옵니다!)
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'dev': '/dev/input/js0'}] # 조이스틱 장치 경로
    )

    # 3. 신호를 변환해주는 joy_teleop 노드
    teleop_node = Node(
        package='f1tenth_fgm_ros2',   # 표준 패키지 사용 시
        executable='joy_teleop.py',
        name='joy_teleop',
        parameters=[parameters_file] # 아까 만든 우리 YAML 파일 적용!
    )

    return LaunchDescription([
        joy_node,
        teleop_node
    ])