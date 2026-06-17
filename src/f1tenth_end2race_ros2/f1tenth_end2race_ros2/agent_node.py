import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
import numpy as np
import torch
import os
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from .model import End2Race

class End2RaceAgent(Node):
    def __init__(self):
        super().__init__('end2race_node')
        
        # 1. 파라미터 선언
        self.declare_parameter('robot_name', 'car1')
        self.declare_parameter('model_path', '/home/f1tenth_ws_foxy/src/f1tenth_end2race_ros2/models/end2race.pth')
        self.declare_parameter('hidden_scale', 4)
        self.declare_parameter('control.max_steer', 0.52)
        self.declare_parameter('control.min_steer', -0.52)
        self.declare_parameter('control.max_speed', 4.0)

        self.robot_name = self.get_parameter('robot_name').value
        self.model_path = self.get_parameter('model_path').value
        self.hidden_scale = self.get_parameter('hidden_scale').value
        self.max_steer = self.get_parameter('control.max_steer').value
        self.min_steer = self.get_parameter('control.min_steer').value
        self.max_speed = self.get_parameter('control.max_speed').value
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = 360 # model.num_features와 일치
        
        # 2. 모델 로드
        self.model = End2Race(hidden_scale=self.hidden_scale).to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.get_logger().info(f"[{self.robot_name}] 모델 로드 성공: {self.model_path}, device: {self.device}")
        else:
            self.get_logger().error(f"[{self.robot_name}] 모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        self.model.eval()
        self.model.set_inference_device(self.device)

        # 3. GRU Hidden State 초기화 (model.py 내부 구조 참조)
        # processed_features = 360 + 360//6 = 420
        # hidden_size = 420 * hidden_scale
        self.hidden_state = None
        self.current_speed = 0.0

        # 최신 라이다 데이터 및 타임스탬프
        self.latest_ranges = None
        self.latest_scan_stamp = None

        # 4. ROS 2 Pub/Sub — ReentrantCallbackGroup으로 scan/drive 병렬 실행
        cb_group = ReentrantCallbackGroup()
        self.scan_sub = self.create_subscription(LaserScan, f'/{self.robot_name}/scan', self.scan_callback, 10, callback_group=cb_group)
self.drive_pub = self.create_publisher(AckermannDriveStamped, f'/{self.robot_name}/drive', 10)

        # 50Hz wall clock 기준 (use_sim_time=True여도 실제 시간으로 발화)
        wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.drive_timer = self.create_timer(1.0 / 100.0, self.drive_callback, clock=wall_clock, callback_group=cb_group)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 30.0, ranges)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)

        if len(ranges) != self.num_features:
            scan_factor = len(ranges) // self.num_features
            n = self.num_features
            blocks = ranges[:n * scan_factor].reshape(n, scan_factor)
            first = ranges[:scan_factor // 2].min()
            last = ranges[-(scan_factor // 2):].min()
            paired = np.lib.stride_tricks.as_strided(
                blocks, shape=(n - 2, 2 * scan_factor),
                strides=(blocks.strides[0], blocks.strides[1])
            )
            ranges = np.concatenate([[first], paired.min(axis=1), [last]])

        self.latest_ranges = ranges
        self.latest_scan_stamp = msg.header.stamp

    def drive_callback(self):
        if self.latest_ranges is None:
            return

        actions, self.hidden_state = self.model.inference_step(
            lidar_data=self.latest_ranges,
            current_speed=self.current_speed,
            prev_hidden=self.hidden_state
        )

        steer_out = float(actions[0])
        speed_out = float(np.clip(actions[1], 0.0, self.max_speed))
        self.current_speed = speed_out

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = np.clip(steer_out, self.min_steer, self.max_steer)
        drive_msg.drive.speed = speed_out
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = End2RaceAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()
