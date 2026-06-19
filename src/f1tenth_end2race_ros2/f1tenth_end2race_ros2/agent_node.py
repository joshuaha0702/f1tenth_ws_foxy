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
        # 학습 시 라이다 FOV (시뮬레이터 270° = ±2.35619 rad 기준).
        # 360개 feature가 이 각도 범위 전체를 균일하게 덮는다고 가정.
        self.declare_parameter('lidar.fov_min', -2.35619)
        self.declare_parameter('lidar.fov_max', 2.35619)
        self.declare_parameter('lidar.max_range', 30.0)

        self.robot_name = self.get_parameter('robot_name').value
        self.model_path = self.get_parameter('model_path').value
        self.hidden_scale = self.get_parameter('hidden_scale').value
        self.max_steer = self.get_parameter('control.max_steer').value
        self.min_steer = self.get_parameter('control.min_steer').value
        self.max_speed = self.get_parameter('control.max_speed').value
        self.fov_min = self.get_parameter('lidar.fov_min').value
        self.fov_max = self.get_parameter('lidar.fov_max').value
        self.lidar_max_range = self.get_parameter('lidar.max_range').value
        
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

        # 각도 기준 다운스케일 매핑 캐시 (라이다 기하가 바뀔 때만 재계산)
        self._scan_cache_key = None
        self._scan_bin_idx = None
        self._scan_valid = None

        # 4. ROS 2 Pub/Sub — ReentrantCallbackGroup으로 scan/drive 병렬 실행
        # robot_name이 비어 있으면 '//scan' 같은 잘못된 토픽이 되므로 안전하게 구성.
        prefix = f'/{self.robot_name.strip("/")}' if self.robot_name and self.robot_name.strip("/") else ''
        scan_topic = f'{prefix}/scan'
        drive_topic = f'{prefix}/drive'
        self.get_logger().info(f"[{self.robot_name}] scan='{scan_topic}', drive='{drive_topic}'")

        cb_group = ReentrantCallbackGroup()
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10, callback_group=cb_group)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # 50Hz wall clock 기준 (use_sim_time=True여도 실제 시간으로 발화)
        wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.drive_timer = self.create_timer(1.0 / 100.0, self.drive_callback, clock=wall_clock, callback_group=cb_group)

    def _build_scan_mapping(self, n_in, angle_min, angle_increment):
        """들어오는 각 빔의 각도를 학습 FOV 기준 360개 bin에 매핑한다.

        라이다 포인트 수나 FOV가 달라도 360개 feature가 항상 학습 시점의
        각도 범위(self.fov_min ~ self.fov_max)를 균일하게 덮도록 보장한다.
        """
        angles = angle_min + np.arange(n_in) * angle_increment
        edges = np.linspace(self.fov_min, self.fov_max, self.num_features + 1)
        # 각 빔이 속하는 bin 인덱스 (FOV 밖 빔은 valid=False로 제외)
        bin_idx = np.digitize(angles, edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < self.num_features)
        self._scan_bin_idx = bin_idx[valid]
        self._scan_valid = valid
        self._scan_cache_key = (n_in, angle_min, angle_increment)

        covered = np.unique(self._scan_bin_idx).size
        if covered < self.num_features:
            self.get_logger().warn(
                f"[{self.robot_name}] 라이다 FOV가 학습 범위를 일부만 덮습니다 "
                f"({covered}/{self.num_features} bin). 빈 bin은 max_range로 채움."
            )

    def scan_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        ranges = np.where(np.isinf(ranges), self.lidar_max_range, ranges)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)

        key = (len(ranges), msg.angle_min, msg.angle_increment)
        if key != self._scan_cache_key:
            self._build_scan_mapping(len(ranges), msg.angle_min, msg.angle_increment)

        # bin별 최솟값(min-pooling)으로 다운스케일 — 가장 가까운 장애물 보존.
        # 빔이 없는 bin(FOV 공백)은 max_range로 남아 '먼 자유공간'으로 취급.
        out = np.full(self.num_features, self.lidar_max_range, dtype=ranges.dtype)
        np.minimum.at(out, self._scan_bin_idx, ranges[self._scan_valid])

        self.latest_ranges = out
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
