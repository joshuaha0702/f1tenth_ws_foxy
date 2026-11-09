import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
import numpy as np
import torch
import os
import time
import gc
import threading
from collections import deque
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
from ackermann_msgs.msg import AckermannDriveStamped
from .model import End2Race, End2RaceNorm, speed_dims_from_state_dict, hidden_scale_from_state_dict
from .scan_preprocess import build_scan_mapping, pool_scan, SpatioTemporalScanFilter

class End2RaceAgent(Node):
    def __init__(self, cli_overrides=None):
        super().__init__('end2race_node')
        
        # 1. 파라미터 선언
        self.declare_parameter('robot_name', 'car1')
        self.declare_parameter('model_path', '/home/f1tenth_ws_foxy/src/f1tenth_end2race_ros2/models/end2race.pth')
        self.declare_parameter('hidden_scale', 4.0)
        self.declare_parameter('control.max_steer', 0.28)
        self.declare_parameter('control.min_steer', -0.28)
        self.declare_parameter('control.max_speed', 2.0)
        self.declare_parameter('control.min_speed', 0.0)
        self.declare_parameter('control.speed_scale', 1.0)
        self.declare_parameter('control.initial_speed', 0.0)
        self.declare_parameter('filter', True)
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
        # train.py --decimate N 모델 호환: N>1 이면 model_rate_hz = 100/N (odom 트리거 100 Hz 기준)
        self.declare_parameter('inference_decimate', 1)
        # 조향 출력 1차 저역통과 시정수 [s]. 0 = 끔. 보간 후 최종 조향에 적용 (LiDAR 전용 모델의 직선 지터 완화용)
        self.declare_parameter('control.steer_lpf_tau', 0.0)
        # 시뮬 전용 실차 재현 파라미터 (0 = 끔). 11/01 실차 bag 에서 측정한 차이를 Gazebo 에 주입해
        # p08 진동의 원인을 가르기 위한 것. 지연은 메시지 stamp(sim time) 기준.
        self.declare_parameter('sim.actuation_delay', 0.0)  # drive 명령 발행 지연 [s] (실차 명령→회전 +~0.11)
        self.declare_parameter('sim.scan_delay', 0.0)       # 추론에 이만큼 오래된 scan 사용 [s] (실차 p08 ~0.1)
        self.declare_parameter('sim.speed_floor', 0.0)      # 0 < 속도 < floor 이면 floor 로 발행 (실차 저속 고정 ~0.66)
        self.declare_parameter('sim.steer_sat', 0.0)        # 발행 조향 = S*tanh(δ/S) (실차 큰 조향 포화 ~0.34)
        self.declare_parameter('sim.steer_clip_min', 0.0)   # VESC servo_max 하드 클립 [rad] (예: 166번 우회전 -0.2918)
        self.declare_parameter('sim.steer_clip_max', 0.0)   # VESC servo_min 하드 클립 [rad] (예: 166번 좌회전 +0.3475)

        # 'timer': wall clock (실차 기본). 'odom': odom 메시지마다 한 스텝 추론해
        # 학습 데이터와 같은 시뮬레이션 시간으로 동작하고 drive stamp = odom stamp.
        self.declare_parameter('trigger', 'timer')
        # 모델의 속도 입력: 'command' = 직전 출력 속도 명령(기존 학습), 'odom' = /<robot>/odom
        # twist.linear.x 측정 속도 (train.py --speed_input odom으로 학습한 모델용),
        # 'hybrid' = [odom 측정 속도, 직전 출력 속도 명령] 2채널 (train.py --speed_input hybrid).
        self.declare_parameter('speed_input', 'command')
        # 전처리(360 bin) scan 을 RViz 확인용으로 발행할지. 추론에는 쓰이지 않는다 (실차 launch 는 use_rviz 에 연동).
        self.declare_parameter('publish_preprocess_scan', True)
        # 타이밍 진단: scan 콜백·drive 틱·프로세스 상태를 1 초 단위로 묶어 /end2race/diag/{scan,tick,sys} 로 발행.
        # scan_qos_depth / legacy_callback_groups 는 수정 전(큐 10, scan·drive 같은 Reentrant 그룹) 재현용.
        self.declare_parameter('diag.timing', False)
        self.declare_parameter('diag.scan_qos_depth', 1)
        self.declare_parameter('diag.legacy_callback_groups', False)

        self.robot_name = self.get_parameter('robot_name').value
        self.model_path = self.get_parameter('model_path').value
        self.hidden_scale = self.get_parameter('hidden_scale').value
        self.max_steer = float(self.get_parameter('control.max_steer').value)
        self.min_steer = float(self.get_parameter('control.min_steer').value)
        self.max_speed = float(self.get_parameter('control.max_speed').value)
        self.min_speed = float(self.get_parameter('control.min_speed').value)

        # CLI 인자 오버라이드 반영
        if cli_overrides:
            if cli_overrides.get('min_speed') is not None:
                self.min_speed = float(cli_overrides['min_speed'])
            if cli_overrides.get('max_speed') is not None:
                self.max_speed = float(cli_overrides['max_speed'])
            if cli_overrides.get('max_steer') is not None:
                self.max_steer = float(cli_overrides['max_steer'])
                self.min_steer = -self.max_steer

        self.get_logger().info(
            f"[{self.robot_name}] 제어 물리 한계: speed=[{self.min_speed:.2f}, {self.max_speed:.2f}] m/s, "
            f"steer=[{self.min_steer:.3f}, {self.max_steer:.3f}] rad"
        )
        self.speed_scale = self.get_parameter('control.speed_scale').value
        self.initial_speed = float(self.get_parameter('control.initial_speed').value)
        self.fov_min = self.get_parameter('lidar.fov_min').value
        self.fov_max = self.get_parameter('lidar.fov_max').value
        self.lidar_max_range = self.get_parameter('lidar.max_range').value
        self.use_norm = self.get_parameter('use_norm').value
        self.drive_rate_hz = float(self.get_parameter('drive_rate_hz').value)
        self.model_rate_hz = float(self.get_parameter('model_rate_hz').value)
        legacy_rate = float(self.get_parameter('rate_hz').value)
        if legacy_rate > 0.0:
            self.model_rate_hz = legacy_rate
        inference_decimate = int(self.get_parameter('inference_decimate').value)
        if inference_decimate > 1:
            self.model_rate_hz = 100.0 / inference_decimate
        self.steer_lpf_tau = float(self.get_parameter('control.steer_lpf_tau').value)
        self._steer_filt = None
        self.sim_actuation_delay = float(self.get_parameter('sim.actuation_delay').value)
        self.sim_scan_delay = float(self.get_parameter('sim.scan_delay').value)
        self.sim_speed_floor = float(self.get_parameter('sim.speed_floor').value)
        self.sim_steer_sat = float(self.get_parameter('sim.steer_sat').value)
        self.sim_steer_clip_min = float(self.get_parameter('sim.steer_clip_min').value)
        self.sim_steer_clip_max = float(self.get_parameter('sim.steer_clip_max').value)
        self._scan_hist = deque(maxlen=200)  # (stamp [s], ranges) — sim.scan_delay 용
        self._act_hist = deque()             # (stamp [s], steer, speed) — sim.actuation_delay 용
        if (self.sim_actuation_delay > 0.0 or self.sim_scan_delay > 0.0 or self.sim_speed_floor > 0.0
                or self.sim_steer_sat > 0.0 or self.sim_steer_clip_min < 0.0 or self.sim_steer_clip_max > 0.0):
            self.get_logger().warn(
                f"[{self.robot_name}] [sim 재현] actuation_delay={self.sim_actuation_delay:.3f}s, "
                f"scan_delay={self.sim_scan_delay:.3f}s, speed_floor={self.sim_speed_floor:.2f} m/s, "
                f"steer_sat={self.sim_steer_sat:.2f} rad, "
                f"steer_clip=[{self.sim_steer_clip_min:.4f}, {self.sim_steer_clip_max:.4f}] rad")
        self.interp_mode = str(self.get_parameter('interp_mode').value).lower()
        if self.interp_mode not in ('linear', 'zoh'):
            raise RuntimeError(f"[{self.robot_name}] interp_mode은 'linear' 또는 'zoh'여야 합니다: {self.interp_mode}")
        self.interp_steps = int(self.get_parameter('interp_steps').value)
        self.trigger = self.get_parameter('trigger').value
        self.speed_input = self.get_parameter('speed_input').value
        # 'none' = 속도 입력 없음, LiDAR 전용 (train.py --speed_input none)
        if self.speed_input not in ('command', 'odom', 'hybrid', 'none'):
            raise RuntimeError(f"[{self.robot_name}] speed_input은 'command', 'odom', 'hybrid', 'none' 중 "
                               f"하나여야 합니다: {self.speed_input}")
        self.uses_odom = self.speed_input in ('odom', 'hybrid')
        self.publish_preprocess_scan = bool(self.get_parameter('publish_preprocess_scan').value)
        self.diag_timing = bool(self.get_parameter('diag.timing').value)
        self.scan_qos_depth = int(self.get_parameter('diag.scan_qos_depth').value)
        self.legacy_callback_groups = bool(self.get_parameter('diag.legacy_callback_groups').value)
        
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
        expected_dims = {'hybrid': 2, 'none': 0}.get(self.speed_input, 1)
        if speed_dims != expected_dims:
            raise RuntimeError(f"[{self.robot_name}] 모델 속도 입력 폭 {speed_dims}이 speed_input="
                               f"'{self.speed_input}'(폭 {expected_dims})과 맞지 않습니다: {self.model_path}")

        # GRU 크기(hidden_scale)도 체크포인트의 gru.weight_hh_l0 에서 읽는다 (파라미터는 참고용)
        ckpt_hidden_scale = hidden_scale_from_state_dict(state)
        if ckpt_hidden_scale != self.hidden_scale:
            self.get_logger().warn(f"[{self.robot_name}] 체크포인트 hidden_scale={ckpt_hidden_scale}, "
                                   f"파라미터 {self.hidden_scale} 무시")
            self.hidden_scale = ckpt_hidden_scale

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
                max_speed=self.max_speed,
                min_speed=self.min_speed
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
        init_v = self.initial_speed if self.initial_speed > 0.0 else (self.min_speed if self.is_norm_model else 0.0)
        self.hidden_state = None
        self.current_speed = init_v
        self.current_steer = 0.0
        self.start_speed = init_v
        self.start_steer = 0.0
        self.target_speed = init_v
        self.target_steer = 0.0
        self.last_inference_speed = init_v  # 모델 speed_input='command'용 직전 추론 속도
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
        self.latest_scan_id = -1

        # 공간-시간 연속성 기반 2중벽 제거 필터
        param_filter = bool(self.get_parameter('filter').value)
        cli_filter = cli_overrides.get('filter') if cli_overrides else None
        self.use_scan_filter = cli_filter if cli_filter is not None else param_filter
        self.get_logger().info(f"[{self.robot_name}] LiDAR 2중벽 시공간 필터 적용: {self.use_scan_filter}")

        self.scan_filter = SpatioTemporalScanFilter(
            num_bins=self.num_features,
            fov_min=self.fov_min,
            fov_max=self.fov_max
        ) if self.use_scan_filter else None

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
        # 추론에 쓴 scan 의 나이 (추론 시각 - scan header stamp) [s]. 11/01 p08 bag 에서 노드가 bag 보다
        # ~100 ms 늦은 scan 으로 추론한 것이 확인되어 실차에서 직접 보이도록 watchdog 주기마다 로그.
        self._scan_age_sum = 0.0
        self._scan_age_max = 0.0
        self._scan_age_n = 0

        # 4. ROS 2 Pub/Sub — scan / drive 콜백을 각자 MutuallyExclusive 그룹에 둔다.
        #    두 그룹끼리는 병렬 실행되고, drive 콜백끼리는 겹치지 않아 hidden_state 를 동시에 읽고 쓰지 않는다.
        prefix = f'/{self.robot_name.strip("/")}' if self.robot_name and self.robot_name.strip("/") else ''
        scan_topic = f'{prefix}/scan'
        drive_topic = f'{prefix}/drive'
        self.get_logger().info(f"[{self.robot_name}] scan='{scan_topic}', drive='{drive_topic}', "
                               f"speed_input='{self.speed_input}'")

        cb_group = ReentrantCallbackGroup()
        if self.legacy_callback_groups:
            scan_group = drive_group = cb_group   # 수정 전 구성 (진단 비교용)
        else:
            scan_group = MutuallyExclusiveCallbackGroup()
            drive_group = MutuallyExclusiveCallbackGroup()
        # 큐 깊이 1(기본): 처리가 밀려도 쌓인 옛 scan 을 차례로 소비하지 않고 항상 최신 scan 만 쓴다.
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, self.scan_qos_depth,
                                                 callback_group=scan_group)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.preprocess_scan_pub = (self.create_publisher(LaserScan, f'{prefix}/preprocess_scan', 10)
                                    if self.publish_preprocess_scan else None)
        self._init_timing_diag()

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
            self.drive_timer = self.create_timer(1.0 / self.drive_rate_hz, self.drive_callback, clock=wall_clock, callback_group=drive_group)

        # 5. 진단 watchdog
        self._watchdog_group = ReentrantCallbackGroup()
        self.watchdog_timer = self.create_timer(
            self._watchdog_period, self._watchdog_callback,
            clock=Clock(clock_type=ClockType.STEADY_TIME), callback_group=self._watchdog_group)

    # ---------------- 타이밍 진단 ----------------
    SCAN_FIELDS = ('scan_id', 't_header', 't_cb_start', 'cb_wait', 'dur_preproc', 'dur_pub', 'dur_total',
                   'drive_active')
    TICK_FIELDS = ('tick', 't_start', 'dt_prev', 'inferred', 'scan_id', 't_scan_header', 'scan_age',
                   'dur_input', 'dur_infer', 'dur_total', 'scan_active', 'drive_active')
    SYS_FIELDS = ('t', 'loadavg1', 'proc_cpu_pct', 'n_threads', 'gc_n', 'gc_total_ms', 'gc_max_ms',
                  'executor_threads', 'scan_qos_depth', 'legacy_groups', 'publish_pre')

    def _init_timing_diag(self):
        self._diag_lock = threading.Lock()
        self._active_scan = 0
        self._active_drive = 0
        self._last_tick_start = None
        self._diag_scan_rows, self._diag_tick_rows = [], []
        self._diag_p1 = 0.0
        self._diag_tls = threading.local()   # 틱별 추론 기록 (수정 전 구성에서 drive 콜백이 겹쳐도 섞이지 않게)
        self._gc_t0 = None
        self._gc_n, self._gc_total, self._gc_max = 0, 0.0, 0.0
        self._proc_cpu_prev = None
        self.executor_threads = 0
        if not self.diag_timing:
            return
        self._diag_pubs = {k: self.create_publisher(Float64MultiArray, f'/end2race/diag/{k}', 10)
                           for k in ('scan', 'tick', 'sys')}
        gc.callbacks.append(self._gc_callback)
        self.create_timer(1.0, self._flush_timing_diag, clock=Clock(clock_type=ClockType.STEADY_TIME),
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f"[{self.robot_name}] [diag] 타이밍 기록 켜짐 → /end2race/diag/{{scan,tick,sys}} "
            f"(scan 큐 {self.scan_qos_depth}, 콜백 그룹 {'수정 전(공유 Reentrant)' if self.legacy_callback_groups else '분리'}, "
            f"preprocess_scan 발행 {'켜짐' if self.publish_preprocess_scan else '꺼짐'})")

    def _ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _gc_callback(self, phase, info):
        if phase == 'start':
            self._gc_t0 = time.perf_counter()
        elif self._gc_t0 is not None:
            d = time.perf_counter() - self._gc_t0
            self._gc_n += 1; self._gc_total += d; self._gc_max = max(self._gc_max, d)
            self._gc_t0 = None

    def _proc_cpu_pct(self):
        try:
            with open('/proc/self/stat') as f:
                fields = f.read().rsplit(')', 1)[1].split()
            ticks = int(fields[11]) + int(fields[12])   # utime + stime
            now = time.monotonic()
            prev, self._proc_cpu_prev = self._proc_cpu_prev, (now, ticks)
            if prev is None:
                return float('nan')
            return 100.0 * (ticks - prev[1]) / os.sysconf('SC_CLK_TCK') / max(now - prev[0], 1e-6)
        except Exception:
            return float('nan')

    def _publish_rows(self, key, fields, rows):
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label='rows', size=len(rows), stride=len(rows) * len(fields)),
                          MultiArrayDimension(label=','.join(fields), size=len(fields), stride=len(fields))]
        msg.data = [float(v) for row in rows for v in row]
        self._diag_pubs[key].publish(msg)

    def _flush_timing_diag(self):
        with self._diag_lock:
            scan_rows, self._diag_scan_rows = self._diag_scan_rows, []
            tick_rows, self._diag_tick_rows = self._diag_tick_rows, []
            gc_stats = (self._gc_n, self._gc_total * 1000.0, self._gc_max * 1000.0)
            self._gc_n, self._gc_total, self._gc_max = 0, 0.0, 0.0
        try:
            load1 = os.getloadavg()[0]
        except OSError:
            load1 = float('nan')
        sys_row = (self._ros_now(), load1, self._proc_cpu_pct(), threading.active_count(), *gc_stats,
                   self.executor_threads, self.scan_qos_depth, float(self.legacy_callback_groups),
                   float(self.publish_preprocess_scan))
        if scan_rows:
            self._publish_rows('scan', self.SCAN_FIELDS, scan_rows)
        if tick_rows:
            self._publish_rows('tick', self.TICK_FIELDS, tick_rows)
        self._publish_rows('sys', self.SYS_FIELDS, [sys_row])

    def _reset_control_state(self):
        """에피소드 정지 및 재시작 시 제어/보간 내부 상태 초기화"""
        init_v = self.initial_speed if self.initial_speed > 0.0 else (self.min_speed if self.is_norm_model else 0.0)
        self.hidden_state = None
        self.current_speed = init_v
        self.current_steer = 0.0
        self.start_speed = init_v
        self.start_steer = 0.0
        self.target_speed = init_v
        self.target_steer = 0.0
        self.last_inference_speed = init_v
        self.tick_counter = 0
        self.interp_step = 0
        self.measured_speed = 0.0
        self._steer_filt = None
        self._scan_hist.clear()
        self._act_hist.clear()
        if getattr(self, 'scan_filter', None) is not None:
            self.scan_filter.reset()

    def _watchdog_callback(self):
        if not self.publishing_enabled:
            return
        if self._scan_age_n > 0:
            self.get_logger().info(
                f"[{self.robot_name}] [diag] 추론 scan 나이 평균 {self._scan_age_sum / self._scan_age_n * 1000.0:.0f} ms, "
                f"최대 {self._scan_age_max * 1000.0:.0f} ms (추론 {self._scan_age_n}회)")
            self._scan_age_sum = 0.0
            self._scan_age_max = 0.0
            self._scan_age_n = 0
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
        if self.diag_timing:
            p0 = time.perf_counter(); t_cb = self._ros_now()
            with self._diag_lock:
                self._active_scan += 1
                drive_active = self._active_drive
        try:
            self._scan_callback(msg)
        finally:
            if self.diag_timing:
                p2 = time.perf_counter()
                t_h = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                with self._diag_lock:
                    self._active_scan -= 1
                    self._diag_scan_rows.append((self.latest_scan_id, t_h, t_cb, t_cb - t_h,
                                                 self._diag_p1 - p0, p2 - self._diag_p1, p2 - p0, drive_active))

    def _scan_callback(self, msg):
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
        if self.use_scan_filter and self.scan_filter is not None:
            out = self.scan_filter(out)

        self.latest_ranges = out
        self.latest_scan_stamp = msg.header.stamp
        self.latest_scan_id += 1
        self._diag_p1 = time.perf_counter()   # 추론에 쓸 수 있게 된 시점 (진단용)
        if self.sim_scan_delay > 0.0:
            self._scan_hist.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, out))
        self._scan_count += 1
        if self._scan_count == 1:
            self.get_logger().info(f"[{self.robot_name}] 첫 scan 수신 ({len(msg.ranges)} beams)")

        # 전처리 결과 RViz 시각화용 LaserScan 발행 (360 bins). 추론에는 쓰이지 않는다.
        if self.preprocess_scan_pub is None:
            return
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
        if not self.diag_timing:
            self._drive_callback(stamp)
            return
        p0 = time.perf_counter(); t_start = self._ros_now()
        with self._diag_lock:
            self._active_drive += 1
            drive_active = self._active_drive - 1
            scan_active = self._active_scan
        self._diag_tls.tick = None
        try:
            self._drive_callback(stamp)
        finally:
            p_end = time.perf_counter()
            dt_prev = t_start - self._last_tick_start if self._last_tick_start is not None else float('nan')
            self._last_tick_start = t_start
            d = self._diag_tls.tick or (0.0, -1.0, float('nan'), float('nan'), float('nan'), float('nan'))
            with self._diag_lock:
                self._active_drive -= 1
                self._diag_tick_rows.append((self.tick_counter, t_start, dt_prev, *d, p_end - p0,
                                             scan_active, drive_active))

    def _drive_callback(self, stamp=None):
        if self.latest_ranges is None or not self.publishing_enabled:
            return

        if stamp is None:
            stamp = self.get_clock().now().to_msg()
        now_s = stamp.sec + stamp.nanosec * 1e-9
        scan_in = self.latest_ranges
        if self.sim_scan_delay > 0.0:
            scan_in = None
            for t, r in reversed(self._scan_hist):
                if t <= now_s - self.sim_scan_delay:
                    scan_in = r
                    break
            if scan_in is None:
                return

        ticks_per_inference = max(1, int(round(self.drive_rate_hz / self.model_rate_hz)))

        # 모델 추론 시점 (예: 100Hz 제어 루프에서 ticks_per_inference 회마다 1번 추론)
        if self.tick_counter % ticks_per_inference == 0:
            if self.speed_input == 'hybrid':
                speed_in = (self.measured_speed, self.last_inference_speed)  # [odom, 직전 모델 출력 속도]
            elif self.speed_input == 'odom':
                speed_in = self.measured_speed
            elif self.speed_input == 'none':
                speed_in = 0.0  # LiDAR 전용 모델은 속도 입력을 쓰지 않음
            else:
                speed_in = self.last_inference_speed

            scan_id_used, scan_stamp_used = self.latest_scan_id, self.latest_scan_stamp
            p_in = time.perf_counter()
            t0 = time.monotonic()
            self._inference_started = t0
            try:
                if self.is_norm_model:
                    actions, self.hidden_state = self.model.inference_step(
                        lidar_data=scan_in,
                        current_speed=speed_in,
                        prev_hidden=self.hidden_state,
                        unnormalize=True
                    )
                else:
                    actions, self.hidden_state = self.model.inference_step(
                        lidar_data=scan_in,
                        current_speed=speed_in,
                        prev_hidden=self.hidden_state
                    )
            finally:
                self._inference_started = None

            latency = time.monotonic() - t0
            if self.diag_timing:
                t_sh = (scan_stamp_used.sec + scan_stamp_used.nanosec * 1e-9) if scan_stamp_used is not None else float('nan')
                # (inferred, scan_id, t_scan_header, scan_age, dur_input, dur_infer)
                self._diag_tls.tick = (1.0, float(scan_id_used), t_sh, self._ros_now() - latency - t_sh,
                                   t0 - p_in, latency)
            self._inference_count += 1
            if self.trigger != 'odom' and self.latest_scan_stamp is not None:
                age = self.get_clock().now().nanoseconds * 1e-9 - (
                    self.latest_scan_stamp.sec + self.latest_scan_stamp.nanosec * 1e-9)
                self._scan_age_sum += age
                self._scan_age_max = max(self._scan_age_max, age)
                self._scan_age_n += 1
            if self._inference_count == 1:
                self.get_logger().info(f"[{self.robot_name}] 첫 추론 완료: {latency * 1000.0:.1f} ms "
                                       f"(device: {self.device})")
            if latency > 1.0 and not self._slow_inference_logged:
                self._slow_inference_logged = True
                self.get_logger().error(f"[{self.robot_name}] inference_step이 {latency:.2f}s 걸렸습니다 "
                                        f"(>1s, device: {self.device}, 추론 {self._inference_count}회째) — "
                                        f"GPU 경합/정지 의심 (이후 동일 경고는 생략)")

            new_steer = float(actions[0])
            lo_speed = self.min_speed if self.is_norm_model else 0.0
            new_speed = float(np.clip(actions[1], lo_speed, self.max_speed)) * self.speed_scale

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

        if self.steer_lpf_tau > 0.0:
            # 발행 주기 기준 1차 저역통과 (odom 트리거는 odom 주기 100 Hz)
            dt = 0.01 if self.trigger == 'odom' else 1.0 / self.drive_rate_hz
            a = dt / (self.steer_lpf_tau + dt)
            self._steer_filt = steer_out if self._steer_filt is None else self._steer_filt + a * (steer_out - self._steer_filt)
            steer_out = self._steer_filt

        steer_pub = float(np.clip(steer_out, self.min_steer, self.max_steer))
        speed_pub = float(speed_out)
        if self.sim_speed_floor > 0.0 and 0.01 < speed_pub < self.sim_speed_floor:
            speed_pub = self.sim_speed_floor
        if self.sim_steer_sat > 0.0:
            steer_pub = float(self.sim_steer_sat * np.tanh(steer_pub / self.sim_steer_sat))
        if self.sim_steer_clip_min < 0.0 or self.sim_steer_clip_max > 0.0:
            lo = self.sim_steer_clip_min if self.sim_steer_clip_min < 0.0 else self.min_steer
            hi = self.sim_steer_clip_max if self.sim_steer_clip_max > 0.0 else self.max_steer
            steer_pub = float(np.clip(steer_pub, lo, hi))
        if self.sim_actuation_delay > 0.0:
            self._act_hist.append((now_s, steer_pub, speed_pub))
            while len(self._act_hist) > 1 and self._act_hist[1][0] <= now_s - self.sim_actuation_delay:
                self._act_hist.popleft()
            t, steer_d, speed_d = self._act_hist[0]
            if t <= now_s - self.sim_actuation_delay:
                steer_pub, speed_pub = (steer_d, speed_d)
            elif self.initial_speed > 0.0:
                steer_pub, speed_pub = (steer_d, self.initial_speed)
            else:
                steer_pub, speed_pub = (0.0, 0.0)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = stamp
        drive_msg.drive.steering_angle = steer_pub
        drive_msg.drive.speed = speed_pub
        self.drive_pub.publish(drive_msg)
        self._last_drive_wall = time.monotonic()

def _parse_cli_args(argv):
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--min_speed', type=float, default=None)
    parser.add_argument('--max_speed', type=float, default=None)
    parser.add_argument('--max_steer', type=float, default=None)
    parsed, remaining = parser.parse_known_args(argv)
    filter_val = None
    if parsed.filter is not None:
        filter_val = parsed.filter.lower() in ('true', '1', 'yes', 't')
    cli_overrides = {
        'filter': filter_val,
        'min_speed': parsed.min_speed,
        'max_speed': parsed.max_speed,
        'max_steer': parsed.max_steer,
    }
    return cli_overrides, remaining

def main(args=None):
    import sys
    raw_args = sys.argv[1:] if args is None else list(args)
    cli_overrides, ros_args = _parse_cli_args(raw_args)
    rclpy.init(args=ros_args)
    node = End2RaceAgent(cli_overrides=cli_overrides)
    executor = MultiThreadedExecutor()
    node.executor_threads = getattr(getattr(executor, '_executor', None), '_max_workers', 0) or 0
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
