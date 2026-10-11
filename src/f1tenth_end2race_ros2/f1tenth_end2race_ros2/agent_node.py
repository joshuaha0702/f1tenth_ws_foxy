import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
import numpy as np
import torch
import os
import time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from .model import End2Race, speed_dims_from_state_dict
from .scan_preprocess import build_scan_mapping, pool_scan

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
        self.declare_parameter('control.speed_scale', 1.0)
        # 학습 시 라이다 FOV (시뮬레이터 270° = ±2.35619 rad 기준).
        # 360개 feature가 이 각도 범위 전체를 균일하게 덮는다고 가정.
        self.declare_parameter('lidar.fov_min', -2.35619)
        self.declare_parameter('lidar.fov_max', 2.35619)
        self.declare_parameter('lidar.max_range', 30.0)
        # 'timer': wall clock 100 Hz (실차 기본). 'odom': odom 메시지마다 한 스텝 추론해
        # 학습 데이터와 같은 시뮬레이션 시간 100 Hz로 동작하고 drive stamp = odom stamp.
        self.declare_parameter('trigger', 'timer')
        # 모델의 속도 입력: 'command' = 직전 출력 속도 명령(기존 학습), 'odom' = /<robot>/odom
        # twist.linear.x 측정 속도 (train.py --speed_input odom으로 학습한 모델용),
        # 'hybrid' = [odom 측정 속도, 직전 출력 속도 명령] 2채널 (train.py --speed_input hybrid).
        self.declare_parameter('speed_input', 'command')

        self.robot_name = self.get_parameter('robot_name').value
        self.model_path = self.get_parameter('model_path').value
        self.hidden_scale = self.get_parameter('hidden_scale').value
        self.max_steer = self.get_parameter('control.max_steer').value
        self.min_steer = self.get_parameter('control.min_steer').value
        self.max_speed = self.get_parameter('control.max_speed').value
        self.speed_scale = self.get_parameter('control.speed_scale').value
        self.fov_min = self.get_parameter('lidar.fov_min').value
        self.fov_max = self.get_parameter('lidar.fov_max').value
        self.lidar_max_range = self.get_parameter('lidar.max_range').value
        self.trigger = self.get_parameter('trigger').value
        self.speed_input = self.get_parameter('speed_input').value
        if self.speed_input not in ('command', 'odom', 'hybrid'):
            raise RuntimeError(f"[{self.robot_name}] speed_input은 'command', 'odom', 'hybrid' 중 하나여야 "
                               f"합니다: {self.speed_input}")
        self.uses_odom = self.speed_input in ('odom', 'hybrid')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = 360 # model.num_features와 일치
        
        # 2. 모델 로드 (속도 입력 폭은 체크포인트의 speed_mlp에서 읽는다)
        # 가중치가 없으면 랜덤 초기화 상태로 주행/수집이 진행되므로 반드시 중단한다.
        if not self.model_path:
            raise RuntimeError(f"[{self.robot_name}] model_path 파라미터가 비어 있습니다 "
                               f"(launch에서 ego_model:=/leader_model:= 로 .pth 경로를 지정하세요)")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"[{self.robot_name}] 모델 파일을 찾을 수 없습니다: {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)
        speed_dims = speed_dims_from_state_dict(state)
        expected_dims = 2 if self.speed_input == 'hybrid' else 1
        if speed_dims != expected_dims:
            raise RuntimeError(f"[{self.robot_name}] 모델 속도 입력 폭 {speed_dims}이 speed_input="
                               f"'{self.speed_input}'(폭 {expected_dims})과 맞지 않습니다: {self.model_path}")
        self.model = End2Race(hidden_scale=self.hidden_scale, speed_dims=speed_dims).to(self.device)
        self.model.load_state_dict(state)
        self.get_logger().info(f"[{self.robot_name}] 모델 로드 성공: {self.model_path}, device: {self.device}")
        
        self.model.eval()
        self.model.set_inference_device(self.device)

        # 3. GRU Hidden State 초기화 (model.py 내부 구조 참조)
        # processed_features = 360 + 360//6 = 420
        # hidden_size = 420 * hidden_scale
        self.hidden_state = None
        self.current_speed = 0.0
        self.measured_speed = 0.0  # speed_input='odom'/'hybrid'일 때 최신 odom twist.linear.x

        # 최신 라이다 데이터 및 타임스탬프
        self.latest_ranges = None
        self.latest_scan_stamp = None

        # 각도 기준 다운스케일 매핑 캐시 (라이다 기하가 바뀔 때만 재계산)
        self._scan_cache = {}

        # 진단용 상태 (동작에는 영향 없음). 시간은 모두 wall monotonic [s].
        # 노드가 살아 있는데 drive를 한 번도 내보내지 않는 경우(scan 미수신, odom 미수신,
        # 추론 정지)를 로그로 구분하기 위한 카운터/타임스탬프.
        self._diag_start = time.monotonic()
        self._scan_count = 0
        self._odom_count = 0
        self._inference_count = 0
        self._last_drive_wall = None       # 마지막 drive publish 시각 (정지 명령 포함)
        self._inference_started = None     # 진행 중인 inference_step 시작 시각 (없으면 None)
        self._slow_inference_logged = False
        self._watchdog_period = 5.0

        # 4. ROS 2 Pub/Sub — ReentrantCallbackGroup으로 scan/drive 병렬 실행
        # robot_name이 비어 있으면 '//scan' 같은 잘못된 토픽이 되므로 안전하게 구성.
        prefix = f'/{self.robot_name.strip("/")}' if self.robot_name and self.robot_name.strip("/") else ''
        scan_topic = f'{prefix}/scan'
        drive_topic = f'{prefix}/drive'
        self.get_logger().info(f"[{self.robot_name}] scan='{scan_topic}', drive='{drive_topic}', "
                               f"speed_input='{self.speed_input}'")

        cb_group = ReentrantCallbackGroup()
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10, callback_group=cb_group)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # EpisodeManager의 /episode/control(START/STOP)을 따라 일시정지하고,
        # STOP마다 GRU hidden state와 이전 속도를 초기화해 에피소드를 독립적으로 만든다.
        # odom 추론과 START/STOP은 hidden state를 공유하므로 한 번에 하나씩만 실행한다.
        self.publishing_enabled = True
        self.stop_pending = False
        # EpisodeManager는 같은 명령을 두 번 보낸다. publishing_enabled로 중복을 판별하면
        # (디스커버리 지연 등으로) 첫 수신 명령이 START일 때 초기화가 건너뛰어지므로
        # 마지막으로 처리한 명령 자체를 기억한다.
        self._last_command = None
        step_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, '/episode/control', self._episode_control_callback, 10,
                                 callback_group=step_group)

        if self.trigger == 'odom':
            self.create_subscription(Odometry, f'{prefix}/odom', self._odom_callback, 20,
                                     callback_group=step_group)
            self.get_logger().info(f"[{self.robot_name}] odom-triggered inference ({prefix}/odom)")
        else:
            if self.uses_odom:
                # timer 트리거: 최신 odom 속도만 받아 두고 다음 타이머 스텝에서 사용
                self.create_subscription(Odometry, f'{prefix}/odom', self._odom_speed_callback, 20,
                                         callback_group=cb_group)
            # 100Hz wall clock 기준 (use_sim_time=True여도 실제 시간으로 발화)
            wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.drive_timer = self.create_timer(1.0 / 100.0, self.drive_callback, clock=wall_clock, callback_group=cb_group)

        # 5. 진단 watchdog — 추론 그룹(step_group)이 막혀도 돌 수 있도록 별도 Reentrant 그룹,
        # use_sim_time과 무관하게 steady clock 5초 주기.
        self._watchdog_group = ReentrantCallbackGroup()
        self.watchdog_timer = self.create_timer(
            self._watchdog_period, self._watchdog_callback,
            clock=Clock(clock_type=ClockType.STEADY_TIME), callback_group=self._watchdog_group)

    def _watchdog_callback(self):
        if not self.publishing_enabled:
            return
        now = time.monotonic()
        no_scan = self._scan_count == 0
        drive_stale = (self._last_drive_wall is None
                       or now - self._last_drive_wall > self._watchdog_period)
        if not (no_scan or drive_stale):
            return
        missing = []
        if no_scan:
            missing.append('scan(미수신)')
        elif self.latest_ranges is None:
            missing.append('scan(START 이후 미수신)')
        if (self.trigger == 'odom' or self.uses_odom) and self._odom_count == 0:
            missing.append('odom(미수신)')
        if self._inference_started is not None:
            missing.append(f'inference(진행 중 {now - self._inference_started:.1f}s — 멈춤 의심)')
        elif self._inference_count == 0:
            missing.append('inference(한 번도 완료 안 됨)')
        last_drive = ('없음' if self._last_drive_wall is None
                      else f'{now - self._last_drive_wall:.1f}s 전')
        self.get_logger().warn(
            f"[{self.robot_name}] [watchdog] publishing 활성인데 drive 정지 — 마지막 drive: {last_drive}, "
            f"누락: {', '.join(missing) if missing else '없음(원인 불명)'} "
            f"(scan {self._scan_count}개, odom {self._odom_count}개, 추론 {self._inference_count}회, "
            f"가동 {now - self._diag_start:.0f}s)")

    def scan_callback(self, msg):
        # 학습 CSV(extract_bag_csv.py)와 같은 함수로 360개 feature를 만든다.
        key = (len(msg.ranges), msg.angle_min, msg.angle_increment)
        if key not in self._scan_cache:
            mapping = build_scan_mapping(len(msg.ranges), msg.angle_min, msg.angle_increment,
                                         self.fov_min, self.fov_max, self.num_features)
            if mapping[2] < self.num_features:
                self.get_logger().warn(
                    f"[{self.robot_name}] 라이다 FOV가 학습 범위를 일부만 덮습니다 "
                    f"({mapping[2]}/{self.num_features} bin). 빈 bin은 max_range로 채움."
                )
            self._scan_cache[key] = mapping
        out = pool_scan(msg.ranges, self._scan_cache[key], self.num_features, self.lidar_max_range)

        self.latest_ranges = out
        self.latest_scan_stamp = msg.header.stamp
        self._scan_count += 1
        if self._scan_count == 1:
            self.get_logger().info(f"[{self.robot_name}] 첫 scan 수신 ({len(msg.ranges)} beams)")

    def _episode_control_callback(self, msg):
        command = msg.data.strip().upper()
        if command not in ('START', 'STOP') or command == self._last_command:
            return
        self._last_command = command
        if command == 'STOP':
            self.publishing_enabled = False
            self.hidden_state = None
            self.current_speed = 0.0
            self.measured_speed = 0.0
            if self.trigger == 'odom':
                # 정지 명령 stamp를 STOP 직후 첫 odom 시각으로 맞춘다 (Pure Pursuit과 동일).
                self.stop_pending = True
            else:
                # timer 트리거에서는 여기서 바로 0 명령을 보내야 차가 마지막 명령을 유지하지 않는다.
                self._publish_stop(self.get_clock().now().to_msg())
        elif command == 'START':
            self.stop_pending = False
            self.hidden_state = None
            self.current_speed = 0.0
            self.measured_speed = 0.0
            # 텔레포트 이전의 scan으로 첫 스텝을 추론하지 않도록 새 scan을 기다린다.
            self.latest_ranges = None
            self.publishing_enabled = True

    def _publish_stop(self, stamp):
        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = stamp
        self.drive_pub.publish(stop_msg)
        self._last_drive_wall = time.monotonic()

    def _count_odom(self):
        self._odom_count += 1
        if self._odom_count == 1:
            self.get_logger().info(f"[{self.robot_name}] 첫 odom 수신")

    def _odom_callback(self, msg):
        self._count_odom()
        if self.stop_pending:
            # 정지 명령은 STOP 직후 첫 odom 시각으로 한 번만 보낸다.
            self.stop_pending = False
            self._publish_stop(msg.header.stamp)
            return
        # 이 스텝을 발화시킨 odom의 측정 속도 (학습 CSV odom_speed = drive stamp 시각의 odom과 동일)
        self.measured_speed = float(msg.twist.twist.linear.x)
        self.drive_callback(stamp=msg.header.stamp)

    def _odom_speed_callback(self, msg):
        self._count_odom()
        self.measured_speed = float(msg.twist.twist.linear.x)

    def drive_callback(self, stamp=None):
        if self.latest_ranges is None or not self.publishing_enabled:
            return

        if self.speed_input == 'hybrid':
            speed_in = (self.measured_speed, self.current_speed)  # [odom, 직전 자기 출력]
        elif self.speed_input == 'odom':
            speed_in = self.measured_speed
        else:
            speed_in = self.current_speed
        # 추론 wall time 측정 (watchdog이 진행 중인 추론의 멈춤을 볼 수 있도록 시작 시각 기록)
        t0 = time.monotonic()
        self._inference_started = t0
        try:
            actions, self.hidden_state = self.model.inference_step(
                lidar_data=self.latest_ranges,
                current_speed=speed_in,
                prev_hidden=self.hidden_state
            )
        finally:
            self._inference_started = None
        latency = time.monotonic() - t0
        self._inference_count += 1
        if self._inference_count == 1:
            self.get_logger().info(f"[{self.robot_name}] 첫 추론 완료: {latency * 1000.0:.1f} ms "
                                   f"(device: {self.device})")
        if latency > 1.0 and not self._slow_inference_logged:
            self._slow_inference_logged = True
            self.get_logger().error(f"[{self.robot_name}] inference_step이 {latency:.2f}s 걸렸습니다 "
                                    f"(>1s, device: {self.device}, 추론 {self._inference_count}회째) — "
                                    f"GPU 경합/정지 의심 (이후 동일 경고는 생략)")

        steer_out = float(actions[0])
        speed_out = float(np.clip(actions[1], 0.0, self.max_speed))
        speed_out *= self.speed_scale
        self.current_speed = speed_out

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = np.clip(steer_out, self.min_steer, self.max_steer)
        drive_msg.drive.speed = speed_out
        self.drive_pub.publish(drive_msg)
        self._last_drive_wall = time.monotonic()

def main(args=None):
    rclpy.init(args=args)
    node = End2RaceAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()
