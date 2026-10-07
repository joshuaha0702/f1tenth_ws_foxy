#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""에피소드 단위 데이터 수집 매니저.

각 에피소드마다:
  1) planner를 STOP 시키고 마지막 0-cmd를 보내 Gazebo 액추에이터를 멈춤
  2) Gazebo physics를 pause 한 뒤 SetEntityState로 차량(들)을 텔레포트
  3) unpause 후 settle_time (sim 시간) 대기
  4) 별도 `ros2 bag record` 프로세스를 띄움 → 녹화 토픽을 전부 구독할 때까지 대기
  5) planner를 START → 정확히 이 시점부터 sim 시간 기반으로 sequence_duration 동안 진행
  6) 진행 중 /car1/chassis_bumper_states, /car2/chassis_bumper_states 를 감시
  7) 시퀀스 길이 도달 시 STOP → bag 종료 → episode_info.yaml(t_start/t_end) 기록 →
     충돌 여부에 따라 clean/ 또는 collision/ 으로 이동
"""

import os
import sys
import math
import time
import random
import shutil
import signal
import threading
import sqlite3
import subprocess
from datetime import datetime

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty as EmptySrv
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType


def yaw_deg_to_quat(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class EpisodeManagerNode(Node):
    def __init__(self):
        super().__init__('episode_manager')

        self.declare_parameter('episodes_yaml', '')
        self.declare_parameter('head2head', False)
        self.declare_parameter('record', False)
        # car1이 lattice planner가 아닐 때(예: End2Race 모델 평가) false.
        # car1 planner의 SetParameters 서비스를 기다리거나 호출하지 않는다.
        self.declare_parameter('car1_planner', True)
        # car2(선두차)도 같은 규칙: End2Race 선두차 모델 평가 시 false.
        self.declare_parameter('car2_planner', True)

        yaml_path = self.get_parameter('episodes_yaml').value
        self.head2head = bool(self.get_parameter('head2head').value)
        record_param = self.get_parameter('record').value
        self.do_record = str(record_param).lower() == 'true'
        self.car1_planner = str(self.get_parameter('car1_planner').value).lower() == 'true'
        self.car2_planner = str(self.get_parameter('car2_planner').value).lower() == 'true'

        # yaml 로드 + 엄격 검증. 실패 시 fatal 로그 후 즉시 종료(비정상 exit code).
        self.cfg = self._load_and_validate_yaml(yaml_path)

        self.sequence_duration_sec = float(self.cfg.get('sequence_duration_sec', 20.0))
        self.settle_time_sec = float(self.cfg.get('settle_time_sec', 1.0))
        out = self.cfg.get('output', {})
        self.base_dir = os.path.expanduser(out.get('base_dir', '/root/f1tenth_ws/bags'))
        self.staging_dir = os.path.join(self.base_dir, out.get('staging_subdir', 'staging'))
        self.clean_dir = os.path.join(self.base_dir, out.get('clean_subdir', 'clean'))
        self.collision_dir = os.path.join(self.base_dir, out.get('collision_subdir', 'collision'))
        for d in (self.staging_dir, self.clean_dir, self.collision_dir):
            os.makedirs(d, exist_ok=True)

        # 녹화 토픽: `-a`로 marker/tf/link_states까지 쓰면 Foxy 레코더가 밀려서
        # drive/odom/scan이 수신 큐에서 버려짐(에피소드 중간 수백 ms 공백).
        # 학습(scan/drive/clock)·검증(odom)·평가에 쓰는 토픽만 녹화한다.
        cars = ['car1', 'car2'] if self.head2head else ['car1']
        default_topics = ['/clock', '/episode/control'] + [
            f'/{car}/{kind}' for car in cars for kind in ('drive', 'odom', 'scan')
        ]
        self.record_topics = list(out.get('record_topics', default_topics))
        # 레코더가 잠깐 밀려도 버티도록 수신 큐를 깊게 잡는다. 발행 측 QoS가 모두
        # reliable/volatile이라 맞춰줌. --max-cache-size는 쓰지 않음: Foxy(rosbag2
        # 0.3.11)는 종료 시 캐시에 남은 메시지를 버려 에피소드 끝 ~2 s가 사라짐.
        record_queue_depth = int(out.get('record_queue_depth', 1000))
        # bag이 에피소드를 다 담지 못하면 같은 초기조건으로 재실행하는 횟수
        self.record_retries = int(out.get('record_retries', 2))
        self.record_qos_path = os.path.join(self.staging_dir, 'record_qos.yaml')
        with open(self.record_qos_path, 'w') as f:
            yaml.safe_dump({
                topic: {'history': 'keep_last', 'depth': record_queue_depth,
                        'reliability': 'reliable', 'durability': 'volatile'}
                for topic in self.record_topics
            }, f)

        # raceline에서 행을 랜덤 샘플링해 에피소드 초기 자세를 자동 생성함
        # (직접 좌표 박는 옛 방식 대신, num_episodes + raceline_path + opponent_offset 으로 설정)
        self.episodes = self._build_episodes_from_config()
        if not self.episodes:
            self.get_logger().fatal('failed to build episode list')
            raise RuntimeError('No episodes to run')

        # Publishers
        self.control_pub = self.create_publisher(String, '/episode/control', 10)

        # 차량별 lattice_planner 노드의 SetParameters 서비스 클라이언트
        # (traj_v_scale 등 동적 파라미터 변경용)
        self.param_cli_car1 = self.create_client(
            SetParameters, '/car1/lattice_planner/set_parameters'
        )
        self.param_cli_car2 = self.create_client(
            SetParameters, '/car2/lattice_planner/set_parameters'
        )

        # Sim clock
        self.sim_time_sec = None
        clock_qos = QoSProfile(depth=10)
        clock_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Clock, '/clock', self._clock_callback, clock_qos)

        # 충돌 감지: 에피소드 진행 중에만 True로 토글되는 플래그
        self._monitoring_collision = False
        self._collision_seen = False
        self._collision_source = None  # 'car1' / 'car2'

        # /car1/drive 발행 재개 시점을 정확히 잡기 위한 구독.
        # START 직후 planner가 다음 timer tick에서 첫 drive를 쏘는 순간을 t_start로 박음.
        self._waiting_for_first_drive = False
        self._first_drive_sim_time = None
        self.create_subscription(
            AckermannDriveStamped, '/car1/drive',
            self._drive_callback, 10
        )
        # car2는 planner timer 위상 차로 car1보다 수십 ms 늦게 출발할 수 있어
        # bag 커버리지 검증용으로 첫 drive 시각을 따로 잡아둔다.
        self._car2_first_drive_sim_time = None
        self._watching_car2_first_drive = False
        if self.head2head:
            self.create_subscription(
                AckermannDriveStamped, '/car2/drive',
                self._car2_drive_callback, 10
            )

        bumper_qos = QoSProfile(depth=20)
        bumper_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            ContactsState, '/car1/chassis_bumper_states',
            lambda m: self._bumper_callback('car1', m), bumper_qos
        )
        if self.head2head:
            self.create_subscription(
                ContactsState, '/car2/chassis_bumper_states',
                lambda m: self._bumper_callback('car2', m), bumper_qos
            )

        # Gazebo services.
        # 주의: simple.world의 gazebo_ros_state 플러그인이 <namespace>/gazebo</namespace>로 박아둬서
        # set_entity_state만 /gazebo/ prefix가 붙고, pause/unpause는 gazebo_ros_init 기본 네임스페이스라
        # prefix 없음. 셋이 비대칭임.
        self.set_state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.pause_cli = self.create_client(EmptySrv, '/pause_physics')
        self.unpause_cli = self.create_client(EmptySrv, '/unpause_physics')

        self.get_logger().info(
            f'EpisodeManager loaded: {len(self.episodes)} episodes, '
            f'duration={self.sequence_duration_sec}s sim, head2head={self.head2head}, '
            f'record={self.do_record}'
        )

        # 에피소드 루프는 별도 스레드에서 돌림.
        # 메인 스레드는 rclpy.spin(node)로 콜백만 처리하고,
        # 이 워커는 future를 polling 방식으로 기다림 (executor 재진입 데드락 방지).
        self._episode_thread = threading.Thread(
            target=self._episode_thread_main, daemon=True, name='episode-loop'
        )
        self._episode_thread.start()

    # ---------------- callbacks ----------------

    def _clock_callback(self, msg: Clock):
        self.sim_time_sec = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _bumper_callback(self, who: str, msg: ContactsState):
        if not self._monitoring_collision:
            return
        if len(msg.states) > 0:
            if not self._collision_seen:
                self._collision_source = who
                names = []
                for st in msg.states[:2]:
                    names.append(f'{st.collision1_name} <-> {st.collision2_name}')
                self.get_logger().warn(f'[collision] {who}: {"; ".join(names)}')
            self._collision_seen = True

    def _drive_callback(self, msg: AckermannDriveStamped):
        if self._waiting_for_first_drive and self._first_drive_sim_time is None:
            # planner_node가 header.stamp를 latest odom stamp로 박아두므로
            # 이게 가장 정확한 "drive가 다시 발행된 순간"
            stamp = msg.header.stamp
            self._first_drive_sim_time = stamp.sec + stamp.nanosec * 1e-9

    def _car2_drive_callback(self, msg: AckermannDriveStamped):
        if self._watching_car2_first_drive and self._car2_first_drive_sim_time is None:
            stamp = msg.header.stamp
            self._car2_first_drive_sim_time = stamp.sec + stamp.nanosec * 1e-9

    # ---------------- yaml validation ----------------

    def _yaml_die(self, msg: str):
        """fatal 로그 후 프로세스 종료 (ros2 launch가 비정상 종료를 감지하도록)."""
        self.get_logger().fatal(f'[episodes_yaml] {msg}')
        # rclpy 정리 후 비정상 exit. SystemExit가 main() finally까지 흘러감.
        raise SystemExit(2)

    def _load_and_validate_yaml(self, yaml_path: str) -> dict:
        if not yaml_path:
            self._yaml_die(
                'episodes_yaml parameter is empty. '
                'Pass episodes:=<path> when launching.'
            )
        if not os.path.isfile(yaml_path):
            self._yaml_die(f'file not found: {yaml_path!r}')

        try:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self._yaml_die(f'YAML parse error in {yaml_path!r}: {e}')
        except OSError as e:
            self._yaml_die(f'cannot read {yaml_path!r}: {e}')

        if cfg is None:
            self._yaml_die(f'{yaml_path!r} is empty')
        if not isinstance(cfg, dict):
            self._yaml_die(
                f'top-level YAML must be a mapping, got {type(cfg).__name__}'
            )

        # ---- 필수 키 ----
        for key in ('num_episodes', 'raceline_path', 'sequence_duration_sec'):
            if key not in cfg:
                self._yaml_die(f'missing required key: {key!r}')

        # ---- num_episodes ----
        try:
            n_ep = int(cfg['num_episodes'])
        except (TypeError, ValueError):
            self._yaml_die(
                f'num_episodes must be an integer, got {cfg["num_episodes"]!r}'
            )
        if n_ep <= 0:
            self._yaml_die(f'num_episodes must be > 0, got {n_ep}')

        # ---- raceline_path ----
        rp = cfg['raceline_path']
        if not isinstance(rp, str) or not rp:
            self._yaml_die(
                f'raceline_path must be a non-empty string, got {rp!r}'
            )
        if not os.path.isfile(rp):
            self._yaml_die(f'raceline_path file not found: {rp!r}')

        # ---- sequence_duration_sec ----
        try:
            dur = float(cfg['sequence_duration_sec'])
        except (TypeError, ValueError):
            self._yaml_die(
                f'sequence_duration_sec must be a number, '
                f'got {cfg["sequence_duration_sec"]!r}'
            )
        if dur <= 0:
            self._yaml_die(f'sequence_duration_sec must be > 0, got {dur}')

        # ---- opponent_offset (필수) ----
        opp = cfg.get('opponent_offset')
        if not isinstance(opp, dict):
            self._yaml_die(
                f'opponent_offset must be a mapping with min/max, got {opp!r}'
            )
        if 'min' not in opp or 'max' not in opp:
            self._yaml_die("opponent_offset must have 'min' and 'max' keys")
        try:
            omin = int(opp['min'])
            omax = int(opp['max'])
        except (TypeError, ValueError):
            self._yaml_die('opponent_offset.min/max must be integers')
        if omin < 1 or omax < omin:
            self._yaml_die(
                f'opponent_offset: require 1 <= min <= max, '
                f'got min={omin}, max={omax}'
            )

        # ---- traj_v_scale (옵션) ----
        v_cfg = cfg.get('traj_v_scale')
        if v_cfg is not None:
            if not isinstance(v_cfg, dict):
                self._yaml_die(
                    f'traj_v_scale must be a mapping or null, '
                    f'got {type(v_cfg).__name__}'
                )
            for who in ('car1', 'car2'):
                sub = v_cfg.get(who)
                if sub is None:
                    continue
                if not isinstance(sub, dict) or 'min' not in sub or 'max' not in sub:
                    self._yaml_die(f'traj_v_scale.{who} must have min/max keys')
                try:
                    vmin = float(sub['min'])
                    vmax = float(sub['max'])
                except (TypeError, ValueError):
                    self._yaml_die(f'traj_v_scale.{who}.min/max must be numbers')
                if vmin <= 0 or vmax < vmin:
                    self._yaml_die(
                        f'traj_v_scale.{who}: require 0 < min <= max, '
                        f'got min={vmin}, max={vmax}'
                    )

        # ---- output (옵션이지만 dict여야 함) ----
        out = cfg.get('output', {})
        if not isinstance(out, dict):
            self._yaml_die(f'output must be a mapping, got {type(out).__name__}')

        # ---- random_seed (옵션) ----
        seed = cfg.get('random_seed')
        if seed is not None:
            try:
                int(seed)
            except (TypeError, ValueError):
                self._yaml_die(
                    f'random_seed must be an integer or null, got {seed!r}'
                )

        # ---- spawn_idx_ranges (옵션) ----
        ranges = cfg.get('spawn_idx_ranges')
        if ranges is not None:
            if not isinstance(ranges, list) or len(ranges) == 0:
                self._yaml_die(
                    'spawn_idx_ranges must be a non-empty list of {min, max} mappings'
                )
            for i, r in enumerate(ranges):
                if not isinstance(r, dict) or 'min' not in r or 'max' not in r:
                    self._yaml_die(
                        f'spawn_idx_ranges[{i}] must have min and max keys'
                    )
                try:
                    rmin = int(r['min'])
                    rmax = int(r['max'])
                except (TypeError, ValueError):
                    self._yaml_die(f'spawn_idx_ranges[{i}].min/max must be integers')
                if rmin < 0 or rmax < rmin:
                    self._yaml_die(
                        f'spawn_idx_ranges[{i}]: require 0 <= min <= max, '
                        f'got min={rmin}, max={rmax}'
                    )

        # ---- spawn_perturbation (옵션) ----
        perturb = cfg.get('spawn_perturbation')
        if perturb is not None:
            if not isinstance(perturb, dict):
                self._yaml_die('spawn_perturbation must be a mapping or null')
            for key in ('lateral_offset_m', 'yaw_deg'):
                val = perturb.get(key)
                if val is not None:
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        self._yaml_die(f'spawn_perturbation.{key} must be a number')
                    if v < 0:
                        self._yaml_die(f'spawn_perturbation.{key} must be >= 0')

        # ---- settle_time_sec (옵션) ----
        if 'settle_time_sec' in cfg:
            try:
                st = float(cfg['settle_time_sec'])
            except (TypeError, ValueError):
                self._yaml_die(
                    f'settle_time_sec must be a number, '
                    f'got {cfg["settle_time_sec"]!r}'
                )
            if st < 0:
                self._yaml_die(f'settle_time_sec must be >= 0, got {st}')

        self.get_logger().info(
            f'[episodes_yaml] validated OK: {yaml_path}'
        )
        return cfg

    # ---------------- raceline-based episode generation ----------------

    def _load_raceline(self, path: str):
        """raceline CSV를 (x, y, psi_rad) 튜플 리스트로 로드.

        포맷: '# s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2' (세미콜론 구분).
        주석/빈줄은 스킵.
        """
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(';')
                if len(parts) < 4:
                    continue
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    psi = float(parts[3])
                except ValueError:
                    continue
                rows.append((x, y, psi))
        return rows

    def _build_episodes_from_config(self):
        num_episodes = int(self.cfg.get('num_episodes', 0))
        raceline_path = self.cfg.get('raceline_path', '')
        opp_cfg = self.cfg.get('opponent_offset', {}) or {}
        offset_min = int(opp_cfg.get('min', 5))
        offset_max = int(opp_cfg.get('max', 20))
        seed = self.cfg.get('random_seed', None)

        # traj_v_scale 랜덤 범위. 키가 없으면 해당 차량은 planner config 기본값 유지.
        v_cfg = self.cfg.get('traj_v_scale', {}) or {}
        v_cfg_car1 = v_cfg.get('car1')
        v_cfg_car2 = v_cfg.get('car2')

        # 위 _load_and_validate_yaml에서 기본 값/타입/경로는 이미 검증됨.
        # 여기서는 raceline 파일을 실제로 파싱했을 때 행 수가 부족한 경우만 추가로 잡음.
        raceline = self._load_raceline(raceline_path)
        n_rows = len(raceline)
        if n_rows == 0:
            self._yaml_die(f'raceline {raceline_path!r} contains no usable rows')
        if n_rows < offset_max + 1:
            self._yaml_die(
                f'raceline too short ({n_rows} rows) for opponent_offset.max={offset_max}'
            )

        # spawn_idx_ranges: 지정된 구간들의 인덱스를 합쳐 후보 풀 생성.
        # 미지정 시 전체 raceline 사용.
        spawn_ranges = self.cfg.get('spawn_idx_ranges')
        if spawn_ranges:
            ego_pool = []
            for r in spawn_ranges:
                rmin = int(r['min'])
                rmax = int(r['max'])
                if rmax >= n_rows:
                    self._yaml_die(
                        f'spawn_idx_ranges max={rmax} >= raceline length {n_rows}'
                    )
                ego_pool.extend(range(rmin, rmax + 1))
            # 중복 제거 (구간이 겹칠 경우 해당 인덱스가 더 자주 샘플될 수 있으므로)
            ego_pool = sorted(set(ego_pool))
            if not ego_pool:
                self._yaml_die('spawn_idx_ranges resolved to an empty candidate pool')
        else:
            ego_pool = list(range(n_rows))

        perturb_cfg = self.cfg.get('spawn_perturbation', {}) or {}
        lateral_half = float(perturb_cfg.get('lateral_offset_m', 0.0))
        yaw_half_deg = float(perturb_cfg.get('yaw_deg', 0.0))

        rng = random.Random(seed) if seed is not None else random.Random()

        # spawn_idx_ranges 요약 문자열 (로그용)
        if spawn_ranges:
            ranges_str = ', '.join(
                f'[{int(r["min"])},{int(r["max"])}]' for r in spawn_ranges
            )
            pool_desc = f'spawn_idx_ranges={ranges_str} → {len(ego_pool)} candidates'
        else:
            pool_desc = f'all {n_rows} rows'

        self.get_logger().info(
            f'Generating {num_episodes} episodes from raceline ({n_rows} rows), '
            f'ego pool: {pool_desc}, '
            f'opponent offset ∈ [{offset_min}, {offset_max}], seed={seed}, '
            f'perturbation: lateral=uniform(±{lateral_half}m), yaw=uniform(±{yaw_half_deg}deg)'
        )

        def row_to_pose(row):
            x, y, psi = row
            # 중심라인 수직 방향(법선)으로 lateral offset 균등 샘플링
            lat = rng.uniform(-lateral_half, lateral_half) if lateral_half > 0.0 else 0.0
            yaw_delta = rng.uniform(-yaw_half_deg, yaw_half_deg) if yaw_half_deg > 0.0 else 0.0
            # 법선 벡터: heading psi 기준 90도 회전 → (-sin(psi), cos(psi))
            px = x - math.sin(psi) * lat
            py = y + math.cos(psi) * lat
            return (
                {'x': px, 'y': py, 'yaw_deg': math.degrees(psi) + yaw_delta},
                lat,
                yaw_delta,
            )

        def sample_v_scale(cfg):
            if not cfg:
                return None
            return rng.uniform(float(cfg['min']), float(cfg['max']))

        episodes = []
        for _ in range(num_episodes):
            ego_idx = rng.choice(ego_pool)
            offset = rng.randint(offset_min, offset_max)
            opp_idx = (ego_idx + offset) % n_rows
            car1_pose, lat1, yaw1 = row_to_pose(raceline[ego_idx])
            car2_pose, lat2, yaw2 = row_to_pose(raceline[opp_idx])
            ep = {
                'car1': car1_pose,
                'car2': car2_pose,
                '_meta': {
                    'ego_idx': ego_idx, 'opp_idx': opp_idx, 'offset': offset,
                    'car1_lateral_m': round(lat1, 3),
                    'car1_yaw_delta_deg': round(yaw1, 2),
                    'car2_lateral_m': round(lat2, 3),
                    'car2_yaw_delta_deg': round(yaw2, 2),
                },
            }
            v1 = sample_v_scale(v_cfg_car1)
            v2 = sample_v_scale(v_cfg_car2)
            if v1 is not None:
                ep['car1_v_scale'] = v1
            if v2 is not None:
                ep['car2_v_scale'] = v2
            episodes.append(ep)
        return episodes

    # ---------------- main loop ----------------

    def _episode_thread_main(self):
        # 메인 스레드의 rclpy.spin이 시동될 시간을 잠깐 줌
        time.sleep(0.5)
        try:
            self._run_all_episodes()
        except Exception as e:
            self.get_logger().error(f'episode loop crashed: {e}')
        finally:
            self.get_logger().info('All episodes done — shutting down.')
            if rclpy.ok():
                rclpy.shutdown()

    def _wait_future(self, future, timeout_sec: float) -> bool:
        """워커 스레드용 future 대기. 메인 스레드 executor가 결과를 채워줄 때까지 polling."""
        deadline = time.time() + timeout_sec
        while not future.done():
            if not rclpy.ok():
                return False
            if time.time() > deadline:
                return False
            time.sleep(0.01)
        return True

    def _wait_for_services(self, timeout_sec: float = 20.0):
        deadline = time.time() + timeout_sec
        required = [
            (self.set_state_cli, '/gazebo/set_entity_state'),
            (self.pause_cli, '/pause_physics'),
            (self.unpause_cli, '/unpause_physics'),
        ]
        if self.car1_planner:
            required.append((self.param_cli_car1, '/car1/lattice_planner/set_parameters'))
        if self.head2head and self.car2_planner:
            required.append(
                (self.param_cli_car2, '/car2/lattice_planner/set_parameters')
            )
        for cli, name in required:
            remaining = max(0.0, deadline - time.time())
            if not cli.wait_for_service(timeout_sec=remaining):
                raise RuntimeError(f'service unavailable: {name}')

    def _set_traj_v_scale(self, cli, value: float, who: str) -> bool:
        req = SetParameters.Request()
        p = ParameterMsg()
        p.name = 'traj_v_scale'
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_DOUBLE
        p.value.double_value = float(value)
        req.parameters = [p]
        future = cli.call_async(req)
        if not self._wait_future(future, timeout_sec= 20.0):
            self.get_logger().error(f'[{who}] SetParameters traj_v_scale timeout')
            return False
        res = future.result()
        if res is None:
            self.get_logger().error(f'[{who}] SetParameters traj_v_scale no result')
            return False
        if not res.results or not res.results[0].successful:
            reason = res.results[0].reason if res.results else 'no result'
            self.get_logger().error(f'[{who}] SetParameters rejected: {reason}')
            return False
        return True

    def _call_empty(self, cli):
        future = cli.call_async(EmptySrv.Request())
        if not self._wait_future(future, timeout_sec=5.0):
            return False
        return future.result() is not None

    def _set_entity_state(self, name: str, pose: dict):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(pose['x'])
        req.state.pose.position.y = float(pose['y'])
        req.state.pose.position.z = 0.05
        qx, qy, qz, qw = yaw_deg_to_quat(float(pose.get('yaw_deg', 0.0)))
        req.state.pose.orientation.x = qx
        req.state.pose.orientation.y = qy
        req.state.pose.orientation.z = qz
        req.state.pose.orientation.w = qw
        # 잔여 관성 제거
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0
        req.state.reference_frame = 'world'
        future = self.set_state_cli.call_async(req)
        if not self._wait_future(future, timeout_sec=5.0):
            self.get_logger().error(f'set_entity_state timeout for {name}')
            return
        res = future.result()
        if res is None or not res.success:
            self.get_logger().error(f'set_entity_state failed for {name}')

    def _publish_control(self, cmd: str):
        msg = String()
        msg.data = cmd
        # 구독자가 잠깐의 join 지연으로 첫 메시지를 놓치는 경우가 있어 두 번 발행
        self.control_pub.publish(msg)
        time.sleep(0.05)
        self.control_pub.publish(msg)

    def _spin_until_sim_time(self, target_sec: float, max_wall_sec: float = None):
        """sim 시간이 target_sec에 도달할 때까지 polling 대기. max_wall_sec는 안전 가드.

        메인 스레드의 executor가 /clock 콜백으로 self.sim_time_sec을 갱신해주므로
        여기선 spin 호출 없이 변수만 polling함.
        """
        wall_deadline = time.time() + (max_wall_sec if max_wall_sec else 1e9)
        while rclpy.ok():
            if self.sim_time_sec is not None and self.sim_time_sec >= target_sec:
                return True
            if time.time() > wall_deadline:
                return False
            time.sleep(0.02)

    def _spin_for_wall(self, seconds: float):
        """wall-clock 기준 단순 sleep. 콜백은 메인 스레드 executor가 알아서 처리함."""
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            time.sleep(min(0.02, max(0.0, end - time.time())))

    # `ros2 bag record`가 만드는 노드 이름 (Foxy rosbag2 0.3.x 고정값)
    RECORDER_NODE_NAME = '_ros2cli_rosbag2'

    def _recorder_subscribed_topics(self) -> set:
        return {
            topic for topic in self.record_topics
            if any(info.node_name == self.RECORDER_NODE_NAME
                   for info in self.get_subscriptions_info_by_topic(topic))
        }

    def _wait_for_recorder(self, subscribed: bool, timeout_sec: float) -> bool:
        """레코더가 녹화 토픽을 전부 구독할 때까지(subscribed=False면 이전 레코더가
        그래프에서 사라질 때까지) wall 기준 polling. 시간 내 도달하면 True."""
        deadline = time.time() + timeout_sec
        while rclpy.ok():
            n = len(self._recorder_subscribed_topics())
            if (n == len(self.record_topics)) if subscribed else (n == 0):
                return True
            if time.time() > deadline:
                return False
            time.sleep(0.05)
        return False

    def _start_bag_record(self, bag_path: str) -> subprocess.Popen:
        # ros2 bag record -o <path> <topics> 를 별도 프로세스 그룹에서 실행
        # SIGINT을 그룹 전체로 보내려고 setsid 사용 (자식까지 정상 종료시킴)
        os.makedirs(os.path.dirname(bag_path), exist_ok=True)
        env = os.environ.copy()
        proc = subprocess.Popen(
            ['ros2', 'bag', 'record', '-o', bag_path,
             '--qos-profile-overrides-path', self.record_qos_path]
            + self.record_topics,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        return proc

    def _stop_bag_record(self, proc: subprocess.Popen):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn('bag record did not exit on SIGINT — sending SIGTERM')
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5.0)
            except Exception:
                pass

    # ---------------- per-episode flow ----------------

    def _run_all_episodes(self):
        self._wait_for_services()
        self.get_logger().info('Gazebo services ready.')

        # /clock이 들어올 때까지 대기 (use_sim_time 환경에서 필수)
        clock_deadline = time.time() + 10.0
        while rclpy.ok() and self.sim_time_sec is None:
            if time.time() > clock_deadline:
                raise RuntimeError(
                    'No /clock messages received — is Gazebo running with publish_clock?'
                )
            time.sleep(0.05)

        for idx, ep in enumerate(self.episodes):
            for attempt in range(self.record_retries + 1):
                if self._run_one_episode(idx, ep, retry_left=attempt < self.record_retries):
                    break

    def _run_one_episode(self, idx: int, ep: dict, retry_left: bool = False) -> bool:
        """에피소드 1회 실행. bag이 에피소드를 다 담지 못했고 retry_left면 bag을
        버리고 False를 반환(같은 초기조건으로 재실행됨)."""
        ep_label = f'ep{idx:04d}'
        meta = ep.get('_meta', {})
        self.get_logger().info('=' * 60)
        self.get_logger().info(
            f'[{ep_label}] starting — ego_idx={meta.get("ego_idx")}, '
            f'opp_idx={meta.get("opp_idx")}, offset={meta.get("offset")}'
        )
        self.get_logger().info(
            f'[{ep_label}] car1={ep.get("car1")}  '
            f'(lat={meta.get("car1_lateral_m", 0.0):+.3f}m, '
            f'yaw_delta={meta.get("car1_yaw_delta_deg", 0.0):+.2f}deg)'
        )
        self.get_logger().info(
            f'[{ep_label}] car2={ep.get("car2")}  '
            f'(lat={meta.get("car2_lateral_m", 0.0):+.3f}m, '
            f'yaw_delta={meta.get("car2_yaw_delta_deg", 0.0):+.2f}deg)'
        )
        if 'car1_v_scale' in ep or 'car2_v_scale' in ep:
            self.get_logger().info(
                f'[{ep_label}] traj_v_scale: car1={ep.get("car1_v_scale")}, '
                f'car2={ep.get("car2_v_scale")}'
            )

        # 1) planner STOP + 0-cmd 박아넣음
        self._publish_control('STOP')
        self._spin_for_wall(0.3)  # 0-cmd가 ackermann_to_twist 통과해 Gazebo plugin까지 닿게 둠

        # 2) pause + teleport + unpause
        self._call_empty(self.pause_cli)
        car1 = ep.get('car1')
        if car1:
            self._set_entity_state('car1', car1)
        if self.head2head:
            car2 = ep.get('car2')
            if car2:
                self._set_entity_state('car2', car2)
        self._call_empty(self.unpause_cli)

        # 3) settle (sim time)
        if self.sim_time_sec is None:
            self.get_logger().warn(f'[{ep_label}] no clock yet, falling back to wall sleep')
            self._spin_for_wall(self.settle_time_sec)
        else:
            t_settle_end = self.sim_time_sec + self.settle_time_sec
            self._spin_until_sim_time(t_settle_end, max_wall_sec=self.settle_time_sec * 5 + 5)

        # 4) bag record subprocess (record=true 일 때만)
        bag_proc = None
        recorder_ready = None
        if self.do_record:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            staging_bag_path = os.path.join(self.staging_dir, f'{ep_label}_{timestamp}')
            # 이전 에피소드 레코더가 그래프에 남아 있으면 새 레코더 구독으로 착각함
            if not self._wait_for_recorder(subscribed=False, timeout_sec=5.0):
                self.get_logger().warn(f'[{ep_label}] previous bag recorder still in the graph')
            bag_proc = self._start_bag_record(staging_bag_path)
            # 고정 0.5 s 대기로는 레코더가 늦게 붙는 경우(최대 ~3 s sim) 에피소드
            # 앞부분이 녹화되지 않았음 → 모든 녹화 토픽 구독을 확인한 뒤 START.
            wait_start = time.time()
            recorder_ready = self._wait_for_recorder(subscribed=True, timeout_sec=30.0)
            if recorder_ready:
                self.get_logger().info(
                    f'[{ep_label}] bag recorder subscribed to {len(self.record_topics)} topics '
                    f'in {time.time() - wait_start:.2f}s'
                )
            else:
                missing = sorted(set(self.record_topics) - self._recorder_subscribed_topics())
                self.get_logger().warn(
                    f'[{ep_label}] bag recorder not subscribed to {missing} after 30s — starting anyway'
                )

        # 5) traj_v_scale 갱신 (START 전에 박아두면 첫 plan부터 새 값 사용)
        # SetParameters 서비스는 reliable이라 단발 호출로 충분, 응답으로 성공 여부도 확인됨
        if self.car1_planner and 'car1_v_scale' in ep:
            self._set_traj_v_scale(self.param_cli_car1, ep['car1_v_scale'], 'car1')
        if self.head2head and self.car2_planner and 'car2_v_scale' in ep:
            self._set_traj_v_scale(self.param_cli_car2, ep['car2_v_scale'], 'car2')

        # 6) START → 첫 /car1/drive 메시지가 들어오는 순간을 t_start로 박음
        self._first_drive_sim_time = None
        self._waiting_for_first_drive = True
        self._car2_first_drive_sim_time = None
        self._watching_car2_first_drive = True
        self._collision_seen = False
        self._collision_source = None
        self._monitoring_collision = True
        self._publish_control('START')

        # planner timer가 첫 plan을 publish할 때까지 polling.
        # 첫 호출은 Numba JIT 컴파일로 수십 초 걸릴 수 있으므로 넉넉히 60초 허용.
        # 정상 흐름이면 ~수백 ms 안에 들어옴.
        wait_deadline = time.time() + 60.0
        while rclpy.ok() and self._first_drive_sim_time is None:
            if time.time() > wait_deadline:
                break
            time.sleep(0.02)
        self._waiting_for_first_drive = False

        if self._first_drive_sim_time is not None:
            t_start = self._first_drive_sim_time
        else:
            # 60초 동안도 안 들어오면 진짜 문제. 그 때만 한 줄 경고.
            self.get_logger().warn(
                f'[{ep_label}] /car1/drive not received within 60s — using current sim time as t_start'
            )
            t_start = self.sim_time_sec
        t_end = t_start + self.sequence_duration_sec
        self.get_logger().info(
            f'[{ep_label}] recording started — t_start={t_start:.3f} (sim), '
            f't_end={t_end:.3f}'
        )

        # 6) 진행 (sim time 기준)
        self._spin_until_sim_time(
            t_end,
            max_wall_sec=self.sequence_duration_sec * 5 + 10,
        )

        # 7) STOP + bag flush
        self._monitoring_collision = False
        self._watching_car2_first_drive = False
        self._publish_control('STOP')
        if bag_proc is not None:
            self._stop_bag_record(bag_proc)

        # 8) 경로 분리 (record=true 일 때만)
        collided = self._collision_seen
        verdict = f'COLLISION ({self._collision_source})' if collided else 'CLEAN'
        if self.do_record:
            first_drive = {'car1': self._first_drive_sim_time,
                           'car2': self._car2_first_drive_sim_time}
            try:
                problems = self._bag_problems(staging_bag_path, t_start, t_end, first_drive)
            except Exception as e:  # 검사 실패로 수집을 멈추진 않음 (validator가 다시 봄)
                problems = []
                self.get_logger().warn(f'[{ep_label}] bag check failed: {e}')
            if problems and retry_left:
                self.get_logger().warn(
                    f'[{ep_label}] bag incomplete — {"; ".join(problems)} — discarding and retrying'
                )
                shutil.rmtree(staging_bag_path, ignore_errors=True)
                return False
            if problems:
                self.get_logger().warn(f'[{ep_label}] bag incomplete, out of retries — {"; ".join(problems)}')
            # validate_episodes.py가 bag이 [t_start, t_end]를 다 덮는지 확인하는 데 씀
            try:
                with open(os.path.join(staging_bag_path, 'episode_info.yaml'), 'w') as f:
                    yaml.safe_dump({
                        't_start': t_start, 't_end': t_end,
                        'first_drive': first_drive,
                        'verdict': 'collision' if collided else 'clean',
                        'collision_source': self._collision_source,
                        'recorder_ready': recorder_ready,
                    }, f)
            except OSError as e:
                self.get_logger().error(f'[{ep_label}] episode_info.yaml write failed: {e}')
            target_root = self.collision_dir if collided else self.clean_dir
            final_path = os.path.join(target_root, os.path.basename(staging_bag_path))
            try:
                shutil.move(staging_bag_path, final_path)
            except Exception as e:
                self.get_logger().error(f'[{ep_label}] move failed: {e}')
                final_path = staging_bag_path
            self.get_logger().info(f'[{ep_label}] {verdict} -> {final_path}')
        else:
            self.get_logger().info(f'[{ep_label}] {verdict} (no bag recorded)')
        return True

    # 토픽별 허용 오차(s): 시작/끝에서 빠져도 되는 길이, 중간 최대 간격
    BAG_EDGE_TOL = {'drive': 0.02, 'odom': 0.02, 'scan': 0.1}
    BAG_MAX_GAP = {'drive': 0.0205, 'odom': 0.0205, 'scan': 0.1}

    def _bag_problems(self, bag_path, t_start, t_end, first_drive) -> list:
        """녹화된 bag이 [t_start, t_end]를 끊김 없이 담았는지 확인.
        레코더가 그래프상 구독을 마쳐도 DDS 매칭이 수 초 늦거나 아예 안 되는
        경우가 있어(Foxy/Fast-DDS) 실제 저장된 메시지의 header stamp로 판단."""
        dbs = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
        if not dbs:
            return ['no .db3 file']
        con = sqlite3.connect(os.path.join(bag_path, dbs[0]))
        try:
            topics = {name: (tid, typ) for tid, name, typ
                      in con.execute('SELECT id, name, type FROM topics')}
            problems = []
            for topic in self.record_topics:
                kind = topic.rsplit('/', 1)[-1]
                if kind not in self.BAG_EDGE_TOL:
                    continue
                if topic not in topics:
                    problems.append(f'{topic} missing')
                    continue
                tid, typ = topics[topic]
                msg_cls = get_message(typ)
                stamps = []
                for (data,) in con.execute(
                        'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,)):
                    st = deserialize_message(data, msg_cls).header.stamp
                    stamps.append(st.sec + st.nanosec * 1e-9)
                car = topic.strip('/').split('/')[0]
                start = t_start
                if kind == 'drive' and first_drive.get(car) is not None:
                    start = first_drive[car]
                # 끝 STOP 0-cmd 등 t_end 이후 메시지는 범위 밖이라 간격 계산에서 제외
                inside = sorted(t for t in stamps if start - 1e-6 <= t <= t_end + 1e-6)
                tol = self.BAG_EDGE_TOL[kind]
                if not inside:
                    problems.append(f'{topic} empty')
                    continue
                if inside[0] - start > tol:
                    problems.append(f'{topic} head missing {inside[0] - start:.2f}s')
                if t_end - inside[-1] > tol:
                    problems.append(f'{topic} tail missing {t_end - inside[-1]:.2f}s')
                gap = max((b - a for a, b in zip(inside, inside[1:])), default=0.0)
                if gap > self.BAG_MAX_GAP[kind]:
                    problems.append(f'{topic} gap {gap * 1000:.0f}ms')
            return problems
        finally:
            con.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    exit_code = 0
    try:
        node = EpisodeManagerNode()
    except SystemExit as e:
        # _yaml_die / validation 실패 — 메시지는 이미 fatal 로그로 출력됨
        exit_code = int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        print(f'[FATAL] episode_manager failed to start: {e}', file=sys.stderr)
        exit_code = 1

    if node is not None:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                node.destroy_node()
            except Exception:
                pass

    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
