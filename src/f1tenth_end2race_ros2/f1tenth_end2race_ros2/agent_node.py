import rclpy
from rclpy.node import Node
import numpy as np
import torch
import os
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from .model import End2Race

class End2RaceAgent(Node):
    def __init__(self):
        super().__init__('end2race_node')
        
        # 1. 파라미터 선언
        self.declare_parameter('robot_name', 'car1')
        self.declare_parameter('model_path', '/home/f1tenth_ws_foxy/src/f1tenth_end2race_ros2/tools/end2race.pth')
        self.declare_parameter('hidden_scale', 4)
        
        self.robot_name = self.get_parameter('robot_name').value
        self.model_path = self.get_parameter('model_path').value
        self.hidden_scale = self.get_parameter('hidden_scale').value
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = 360 # model.num_features와 일치
        
        # 2. 모델 로드
        self.model = End2Race(hidden_scale=self.hidden_scale).to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.get_logger().info(f"[{self.robot_name}] 모델 로드 성공: {self.model_path}")
        else:
            self.get_logger().error(f"[{self.robot_name}] 모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        self.model.eval()
        
        # 3. GRU Hidden State 초기화 (model.py 내부 구조 참조)
        # processed_features = 360 + 360//6 = 420
        # hidden_size = 420 * hidden_scale
        self.hidden_state = None 
        self.current_speed = 0.0
        
        # 4. ROS 2 Pub/Sub
        self.scan_sub = self.create_subscription(LaserScan, f'/{self.robot_name}/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, f'/{self.robot_name}/drive', 10)

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        # LiDAR 데이터 전처리
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max)
        
        if len(ranges) != self.num_features:
            indices = np.linspace(0, len(ranges)-1, self.num_features, dtype=int)
            ranges = ranges[indices]

        # 5. model.py의 inference_step 활용
        # 이 함수 내부에서 Tensor 변환 및 Device 처리가 수행됨
        actions, self.hidden_state = self.model.inference_step(
            lidar_data=ranges,
            current_speed=self.current_speed,
            prev_hidden=self.hidden_state
        )

        # 결과: actions = [steering, speed]
        steer_out = float(actions[0])
        speed_out = float(actions[1])

        # 6. 제어 메시지 발행
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = np.clip(steer_out, -0.52, 0.52)
        drive_msg.drive.speed = speed_out
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(End2RaceAgent())
    rclpy.shutdown()
