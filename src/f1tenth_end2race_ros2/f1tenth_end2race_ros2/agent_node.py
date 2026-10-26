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
from .model import End2Race, End2RaceNorm, speed_dims_from_state_dict
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
        self.declare_parameter('control.max_speed', 2.0)
        self.declare_parameter('control.speed_scale', 1.0)
        # 학습 시 라이다 FOV (시뮬레이터 270° = ±2.35619 rad 기준).
        # 360개 feature가 이 각도 범위 전체를 균일하게 덮는다고 가정.
        self.declare_parameter('lidar.fov_min', -2.35619)
        self.declare_parameter('lidar.fov_max', 2.35619)
        self.declare_parameter('lidar.max_range', 30.0)

        # 정규화 모델 및 주기/보간 제어 파라미터
        self.declare_parameter('use_norm', 'auto')          # 'auto'(가중치 자동 감지), 'true'(강제 정규화), 'false'(기존)
        self.declare_parameter('drive_rate_hz', 100.0)      # 차량 drive 토픽 발행 주기 (기본 100Hz)
        self.declare_parameter('model_rate_hz', 10.0)       # 모델 추론 주기 (10Hz 다운샘플 모델은 10Hz, 기존 모델은 100Hz)
        self.declare_parameter('interp_mode', 'linear')     # 보간 방식: 'linear' (선형보간) 또는 'zoh' (Zero-Order Hold)
        self.declare_parameter('interp_steps', 4)           # 선형 보간 램프 스텝 수 (기본 4스텝 = 40ms 동안 빠르게 도달 후 나머지 유지)
        self.declare_parameter('rate_hz', 0.0)              # 기존 파라미터 호환용 (0보다 크면 model_rate_hz 대체)

        # 'timer': wall clock (실차 기본). 'odom': odom 메시지마다 한 스텝 추론해
        # 학습 데이터와 같은 시뮬레이션 시간으로 동작하고 drive stamp = odom stamp.
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
        self.use_norm = self.get_parameter('use_norm').value
        self.drive_rate_hz = float(self.get_parameter('drive_rate_hz').value)
        self.model_rate_hz = float(self.get_parameter('model_rate_hz').value)
        legacy_rate = float(self.get_parameter('rate_hz').value)
        if legacy_rate > 0.0:
            self.model_rate_hz = legacy_rate
        self.interp_mode = str(self.get_parameter('interp_mode').value).lower()
        if self.interp_mode not in ('linear', 'zoh'):
            raise RuntimeError(f"[{self.robot_name}] interp_mode은 'linear' 또는 'zoh'여야 합니다: {self.interp_mode}")
        self.interp_steps = int(self.get_parameter('interp_steps').value)
        self.trigger = self.get_parameter('trigger').value
        self.speed_input = self.get_parameter('speed_input').value
        if self.speed_input not in ('command', 'odom', 'hybrid'):
            raise RuntimeError(f"[{self.robot_name}] speed_input은 'command', 'odom', 'hybrid' 중 하나여야 "
                                f"합니다: {self.speed_input}")
        self.uses_odom = self.speed_input in ('odom', 'hybrid')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = 360 # model.num_features와 일치
        
        # 2. 모델 로드 (속도 입력 폭은 체크포인트의 speed_mlp에서 읽는다)
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

        # 정규화 모델 판별 ('auto'인 경우 가중치 내 'feature_norm.weight' 유무로 자동 감지)
        use_norm_str = str(self.use_norm).strip().lower()
        if use_norm_str in ('true', '1'):
            self.is_norm_model = True
        elif use_norm_str in ('false', '0'):
            self.is_norm_model = False
        else:  # 'auto'
            self.is_norm_model = ('feature_norm.weight' in state)

        if self.is_norm_model:
            self.model = End2RaceNorm(
                hidden_scale=self.hidden_scale,
                speed_dims=speed_dims,
                max_steer=self.max_steer,
                max_speed=self.max_speed
            ).to(self.device)
            self.get_logger().info(f"[{self.robot_name}] [정규화 모델] End2RaceNorm 로드 성공: {self.model_path}, device: {self.device}")
        else:
            self.model = End2Race(
                hidden_scale=self.hidden_scale,
                speed_dims=speed_dims
            ).to(self.device)
            self.get_logger().info(f"[{self.robot_name}] [표준 모델] End2Race 로드 성공: {self.model_path}, device: {self.device}")

        self.model.load_state_dict(state)
        self.model.eval()
        self.model.set_inference_device(self.device)

        # 3. GRU Hidden State & 제어 보간 상태 초기화
        self.hidden_state = None
        self.current_speed = 0.0
        self.current_steer = 0.0
        self.start_speed = 0.0
        self.start_steer = 0.0
        self.target_speed = 0.0
        self.target_steer = 0.0
        self.last_inference_speed = 0.0  # 모델 speed_input='command'용 직전 추론 속도
        self.tick_counter = 0
        self.interp_step = 0
        self.measured_speed = 0.0  # speed_input='odom'/'hybrid'일 때 최신 odom twist.linear.x

        ticks_per_infer = max(1, int(round(self.drive_rate_hz / self.model_rate_hz)))
        ramp_steps = min(max(1, self.interp_steps), ticks_per_infer)
        self.get_logger().info(
            f"[{self.robot_name}] Control: drive={self.drive_rate_hz:.1f}Hz, model={self.model_rate_hz:.1f}Hz, "
            f"interp={self.interp_mode} (ramp={ramp_steps} steps / {ramp_steps * (1000.0/self.drive_rate_hz):.0f}ms, "
            f"hold={max(0, ticks_per_infer - ramp_steps) * (1000.0/self.drive_rate_hz):.0f}ms)"
        )

        # 최신 라이다 데이터 및 타임스탬프
        self.latest_ranges = None
        self.latest_scan_stamp = None

        # 각도 기준 다운스케일 매핑 캐시 (라이다 기하가 바뀔 때만 재계산)
        self._scan_cache = {}

        # 진단용 상태 (동작에는 영향 없음). 시간은 모두 wall monotonic [s].
        self._diag_start = time.monotonic()
        self._scan_count = 0
        self._odom_count = 0
        self._inference_count = 0
        self._last_drive_wall = None       # 마지막 drive publish 시각 (정지 명령 포함)
        self._inference_started = None     # 진행 중인 inference_step 시작 시각 (없으면 None)
        self._slow_inference_logged = False
        self._watchdog_period = 5.0

        # 4. ROS 2 Pub/Sub — ReentrantCallbackGroup으로 scan/drive 병렬 실행
        prefix = f'/{self.robot_name.strip("/")}' if self.robot_name and self.robot_name.strip("/") else ''
        scan_topic = f'{prefix}/scan'
        drive_topic = f'{prefix}/drive'
        self.get_logger().info(f"[{self.robot_name}] scan='{scan_topic}', drive='{drive_topic}', "
                               f"speed_input='{self.speed_input}'")

        cb_group = ReentrantCallbackGroup()
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10, callback_group=cb_group)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.preprocess_scan_pub = self.create_publisher(LaserScan, f'{prefix}/preprocess_scan', 10)

        # EpisodeManager의 /episode/control(START/STOP)을 따라 일시정지하고,
        # STOP마다 GRU hidden state와 이전 속도를 초기화해 에피소드를 독립적으로 만든다.
        self.publishing_enabled = True
        self.stop_pending = False
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
            # wall clock 기준 drive_rate_hz (기본 100Hz) 주기로 발화하여 보간된 제어 명령 발행
            wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.drive_timer = self.create_timer(1.0 / self.drive_rate_hz, self.drive_callback, clock=wall_clock, callback_group=cb_group)

        # 5. 진단 watchdog
        self._watchdog_group = ReentrantCallbackGroup()
        self.watchdog_timer = self.create_timer(
            self._watchdog_period, self._watchdog_callback,
            clock=Clock(clock_type=ClockType.STEADY_TIME), callback_group=self._watchdog_group)

    def _reset_control_state(self):
        """에피소드 정지 및 재시작 시 제어/보간 내부 상태 초기화"""
        self.hidden_state = None
        self.current_speed = 0.0
        self.current_steer = 0.0
        self.start_speed = 0.0
        self.start_steer = 0.0
        self.target_speed = 0.0
        self.target_steer = 0.0
        self.last_inference_speed = 0.0
        self.tick_counter = 0
        self.interp_step = 0
        self.measured_speed = 0.0

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

        # 전처리 결과 RViz 시각화용 LaserScan 발행 (360 bins)
        pre_msg = LaserScan()
        pre_msg.header = msg.header
        pre_msg.angle_min = float(self.fov_min)
        pre_msg.angle_max = float(self.fov_max)
        pre_msg.angle_increment = float((self.fov_max - self.fov_min) / self.num_features)
        pre_msg.time_increment = 0.0
        pre_msg.scan_time = msg.scan_time
        pre_msg.range_min = float(msg.range_min)
        pre_msg.range_max = float(self.lidar_max_range)
        pre_msg.ranges = [float(x) for x in out]
        self.preprocess_scan_pub.publish(pre_msg)

    def _episode_control_callback(self, msg):
        command = msg.data.strip().upper()
        if command not in ('START', 'STOP') or command == self._last_command:
            return
        self._last_command = command
        if command == 'STOP':
            self.publishing_enabled = False
            self._reset_control_state()
            if self.trigger == 'odom':
                # 정지 명령 stamp를 STOP 직후 첫 odom 시각으로 맞춘다 (Pure Pursuit과 동일).
                self.stop_pending = True
            else:
                # timer 트리거에서는 여기서 바로 0 명령을 보내야 차가 마지막 명령을 유지하지 않는다.
                self._publish_stop(self.get_clock().now().to_msg())
        elif command == 'START':
            self.stop_pending = False
            self._reset_control_state()
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
            self.stop_pending = False
            self._publish_stop(msg.header.stamp)
            return
        self.measured_speed = float(msg.twist.twist.linear.x)
        self.drive_callback(stamp=msg.header.stamp)

    def _odom_speed_callback(self, msg):
        self._count_odom()
        self.measured_speed = float(msg.twist.twist.linear.x)

    def drive_callback(self, stamp=None):
        if self.latest_ranges is None or not self.publishing_enabled:
            return

        ticks_per_inference = max(1, int(round(self.drive_rate_hz / self.model_rate_hz)))

        # 모델 추론 시점 (예: 100Hz 제어 루프에서 ticks_per_inference 회마다 1번 추론)
        if self.tick_counter % ticks_per_inference == 0:
            if self.speed_input == 'hybrid':
                speed_in = (self.measured_speed, self.last_inference_speed)  # [odom, 직전 모델 출력 속도]
            elif self.speed_input == 'odom':
                speed_in = self.measured_speed
            else:
                speed_in = self.last_inference_speed

            t0 = time.monotonic()
            self._inference_started = t0
            try:
                if self.is_norm_model:
                    actions, self.hidden_state = self.model.inference_step(
                        lidar_data=self.latest_ranges,
                        current_speed=speed_in,
                        prev_hidden=self.hidden_state,
                        unnormalize=True
                    )
                else:
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

            new_steer = float(actions[0])
            new_speed = float(np.clip(actions[1], 0.0, self.max_speed)) * self.speed_scale

            # 보간 구간 갱신: 시작점은 현재 출력값, 목표점은 새 추론값
            self.start_steer = self.current_steer
            self.start_speed = self.current_speed
            self.target_steer = new_steer
            self.target_speed = new_speed
            self.last_inference_speed = new_speed
            self.interp_step = 0

        self.interp_step += 1
        self.tick_counter += 1

        # 보간 계산 (선형 보간 'linear' vs 영차 홀드 'zoh')
        ramp_steps = min(max(1, self.interp_steps), ticks_per_inference)
        if self.interp_mode == 'linear' and ticks_per_inference > 1 and ramp_steps > 0:
            alpha = min(1.0, float(self.interp_step) / float(ramp_steps))
            steer_out = (1.0 - alpha) * self.start_steer + alpha * self.target_steer
            speed_out = (1.0 - alpha) * self.start_speed + alpha * self.target_speed
        else:  # 'zoh' 또는 ticks_per_inference == 1
            steer_out = self.target_steer
            speed_out = self.target_speed

        self.current_steer = steer_out
        self.current_speed = speed_out

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = float(np.clip(steer_out, self.min_steer, self.max_steer))
        drive_msg.drive.speed = float(speed_out)
        self.drive_pub.publish(drive_msg)
        self._last_drive_wall = time.monotonic()

def main(args=None):
    rclpy.init(args=args)
    node = End2RaceAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
