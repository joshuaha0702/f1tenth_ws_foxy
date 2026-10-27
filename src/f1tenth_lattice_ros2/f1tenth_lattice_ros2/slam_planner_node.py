#!/usr/bin/env python3

import os
import glob
import math
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovarianceStamped, Point, TransformStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64MultiArray, MultiArrayDimension

from f1tenth_lattice_ros2.planner_utils import load_config, nearest_point, xy_2_rc
from f1tenth_lattice_ros2.lattice_planner import LatticePlanner
from f1tenth_lattice_ros2.obstacle_tracker import MultiObstacleTracker


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def resolve_map_prefix(map_path):
    """
    폴더 경로 또는 파일 경로가 들어왔을 때, 
    LatticePlanner가 사용할 확장자 제거된 Prefix 경로(예: .../Simple/Simple_map)를 반환
    """
    if os.path.isdir(map_path):
        yamls = glob.glob(os.path.join(map_path, '*.yaml'))
        if not yamls:
            raise FileNotFoundError(f"No .yaml file found inside directory: {map_path}")
        return os.path.splitext(yamls[0])[0]
    
    if map_path.endswith('.yaml'):
        return os.path.splitext(map_path)[0]
        
    return map_path


class SlamLatticePlannerNode(Node):
    def __init__(self):
        super().__init__('slam_lattice_planner')

        # --- Parameters ---
        self.declare_parameter('config_path', '')
        self.declare_parameter('raceline_path', '')
        self.declare_parameter('map_path', '')
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('max_steering_angle', 0.26)
        self.declare_parameter('plan_frequency', 10.0)
        self.declare_parameter('localization_mode', 'scan')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('odom_cov_scale', 100.0)
        self.declare_parameter('overtake_mode', 'none')
        self.declare_parameter('opponent_namespace', 'car2')
        self.declare_parameter('opponent_odom_topic', '')
        self.declare_parameter('enable_overtake', False)
        self.declare_parameter('opponent_weight', -1.0)

        config_path = self.get_parameter('config_path').value
        raceline_path = self.get_parameter('raceline_path').value
        raw_map_path = self.get_parameter('map_path').value
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer = float(self.get_parameter('max_steering_angle').value)
        self.plan_freq = float(self.get_parameter('plan_frequency').value)
        self.localization_mode = str(self.get_parameter('localization_mode').value).lower()
        self.initial_x = float(self.get_parameter('initial_x').value)
        self.initial_y = float(self.get_parameter('initial_y').value)
        self.initial_yaw = float(self.get_parameter('initial_yaw').value)

        # 추월 모드 판정: 'none' (기본값), 'odom' (상대차량 위치 토픽 수신), 'lidar' (라이다 동적 추적)
        raw_mode = str(self.get_parameter('overtake_mode').value).lower()
        self.enable_overtake = bool(self.get_parameter('enable_overtake').value)
        if raw_mode in ['none', 'off', 'false', '0']:
            if self.enable_overtake:
                self.overtake_mode = 'lidar'
            else:
                self.overtake_mode = 'none'
        elif raw_mode in ['odom', 'topic', 'ground_truth']:
            self.overtake_mode = 'odom'
        elif raw_mode in ['lidar', 'scan']:
            self.overtake_mode = 'lidar'
        else:
            self.get_logger().warn(f"Unknown overtake_mode '{raw_mode}', defaulting to 'none'")
            self.overtake_mode = 'none'

        self.opponent_namespace = str(self.get_parameter('opponent_namespace').value)
        self.opponent_odom_topic = str(self.get_parameter('opponent_odom_topic').value)
        self.opponent_weight = float(self.get_parameter('opponent_weight').value)

        if not config_path or not raceline_path or not raw_map_path:
            self.get_logger().fatal('config_path, raceline_path, map_path parameters are required')
            raise RuntimeError('Missing required parameters')

        map_path = resolve_map_prefix(raw_map_path)

        self.get_logger().info(f'Loading config from: {config_path}')
        self.get_logger().info(f'Loading raceline from: {raceline_path}')
        self.get_logger().info(f'Resolved map prefix: {map_path}')

        ns = self.get_namespace().strip('/')
        if not ns:
            ns = 'car1'
        
        conf = load_config(config_path, namespace=ns)
        if not hasattr(conf, 'traj_points'):
            conf = load_config(config_path, namespace='')
            if not hasattr(conf, 'traj_points'):
                conf.traj_points = 10

        self.planner = LatticePlanner(conf, map_path, raceline_path)

        # 파라미터 기반 추월/회피 모드 설정
        if self.overtake_mode == 'none':
            self.planner.cost_weights[-1] = 0.0
            self.get_logger().info('Overtake mode: NONE (Disabled, following raceline only)')
        elif self.overtake_mode == 'odom':
            if self.opponent_weight >= 0.0:
                self.planner.cost_weights[-1] = self.opponent_weight
            self.get_logger().info(f'Overtake mode: ODOM (Tracking opponent via topic, weight={self.planner.cost_weights[-1]})')
        elif self.overtake_mode == 'lidar':
            if self.opponent_weight >= 0.0:
                self.planner.cost_weights[-1] = self.opponent_weight
            self.get_logger().info(f'Overtake mode: LIDAR (Dynamic obstacle tracking via LiDAR, weight={self.planner.cost_weights[-1]})')

        self.get_logger().info('Lattice planner initialized successfully!')

        self.pose_x = self.initial_x
        self.pose_y = self.initial_y
        self.pose_theta = self.initial_yaw
        self.velocity = 0.0
        self.amcl_received = False
        self.odom_received = False

        self.initial_map_x = self.initial_x
        self.initial_map_y = self.initial_y
        self.initial_map_yaw = self.initial_yaw
        self.odom_base_x = None
        self.odom_base_y = None
        self.odom_base_yaw = None
        self.last_odom_x = None
        self.last_odom_y = None
        self.last_odom_yaw = None
        self.map_to_odom_transform = None
        self.last_raw_odom_x = 0.0
        self.last_raw_odom_y = 0.0
        self.last_raw_odom_yaw = 0.0

        self.opp_pose = np.empty((0, 3))
        self.opp_vels = np.empty((0, 2))
        self.obstacle_tracker = MultiObstacleTracker(
            max_match_dist=1.2,
            min_hits_to_confirm=3,
            max_misses_to_delete=3
        )
        self._last_plan_time = None
        self.best_traj = None
        # 계획이 연속으로 이만큼 실패하면 빈 planned_trajectory를 보내 컨트롤러를 즉시 멈춘다
        # (컨트롤러의 trajectory_timeout 보다 먼저, 그리고 명시적으로).
        self.plan_fail_count = 0
        self.plan_fail_stop_threshold = 3

        qos = QoSProfile(depth=10)
        marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        if self.localization_mode in ['scan', 'amcl']:
            self.get_logger().info('Localization mode: SCAN/AMCL ONLY (Map pose from AMCL, Velocity from Odom)')
            self.create_subscription(
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_callback, qos
            )
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._amcl_pose_callback, qos
            )
            self.create_subscription(
                Odometry, 'odom', self._odom_velocity_callback, qos
            )
        elif self.localization_mode == 'fusion':
            self.get_logger().info('Localization mode: FUSION (Map anchor from AMCL + Relative Odom updates)')
            self.create_subscription(
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_fusion_callback, qos
            )
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._amcl_pose_fusion_callback, qos
            )
            self.create_subscription(
                Odometry, 'odom', self._odom_fusion_callback, qos
            )
        else:
            self.get_logger().info('Localization mode: ODOMETRY ONLY (Initial pose from 2D Pose Estimate + Odom updates)')
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._initialpose_callback, qos
            )
            self.create_subscription(
                Odometry, 'odom', self._odom_full_callback, qos
            )

        if self.overtake_mode == 'odom':
            opp_topic = self.opponent_odom_topic if self.opponent_odom_topic else f'/{self.opponent_namespace}/odom'
            self.create_subscription(
                Odometry, opp_topic, self._opp_odom_callback, qos
            )
            self.get_logger().info(f'Subscribed to opponent odom topic: {opp_topic}')

        self.create_subscription(
            LaserScan, 'scan', self._scan_callback, qos_profile_sensor_data
        )

        # Planned trajectory publisher (read by pure_pursuit_controller_node)
        self.trajectory_pub = self.create_publisher(
            Float64MultiArray, 'planned_trajectory', QoSProfile(depth=1)
        )
        self.raceline_pub = self.create_publisher(MarkerArray, 'raceline_marker', marker_qos)
        self.best_traj_pub = self.create_publisher(Marker, 'best_traj_marker', qos)
        self.obstacle_marker_pub = self.create_publisher(MarkerArray, 'obstacle_marker', qos)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Planning Timer
        period = 1.0 / max(self.plan_freq, 1.0)
        self.timer = self.create_timer(period, self._plan_callback)

        # TF Broadcast Timer (50Hz)
        self.tf_timer = self.create_timer(0.02, self._publish_tf_timer)

        # Raceline visualization marker (1s interval)
        self.create_timer(1.0, self._publish_raceline_vis)

        self.get_logger().info(
            f'SLAM lattice planner ready (mode={self.localization_mode}, overtake_mode={self.overtake_mode}, '
            f'plan_freq={self.plan_freq:.1f}Hz, max_speed={self.max_speed}m/s, max_steer={self.max_steer:.3f}rad)'
        )

    def _publish_tf_timer(self):
        if not self.amcl_received or self.map_to_odom_transform is None:
            return

        tx, ty, rel_yaw = self.map_to_odom_transform

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.translation.z = 0.0

        cy = math.cos(rel_yaw * 0.5)
        sy = math.sin(rel_yaw * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(sy)
        t.transform.rotation.w = float(cy)
        self.tf_broadcaster.sendTransform(t)

    def _opp_odom_callback(self, msg: Odometry):
        if self.overtake_mode != 'odom':
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        v = msg.twist.twist.linear.x
        vx = v * math.cos(theta)
        vy = v * math.sin(theta)

        self.opp_pose = np.array([[x, y, theta]], dtype=np.float64)
        self.opp_vels = np.array([[vx, vy]], dtype=np.float64)

        tracked_info = [{
            'id': 1,
            'x': float(x), 'y': float(y), 'theta': float(theta),
            'vx': float(vx), 'vy': float(vy), 'speed': float(abs(v))
        }]
        self._publish_obstacle_markers(tracked_info)

    def _scan_callback(self, msg: LaserScan):
        if self.overtake_mode != 'lidar':
            return

        if not self.amcl_received or self.planner is None:
            return

        detected = self._extract_obstacles_from_scan(msg)
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        tracked = self.obstacle_tracker.update(detected, now_sec)

        if len(tracked) > 0:
            opp_poses = []
            opp_vels = []
            for tr in tracked:
                opp_poses.append([tr['x'], tr['y'], tr['theta']])
                opp_vels.append([tr['vx'], tr['vy']])
            self.opp_pose = np.array(opp_poses, dtype=np.float64)
            self.opp_vels = np.array(opp_vels, dtype=np.float64)
        else:
            self.opp_pose = np.empty((0, 3), dtype=np.float64)
            self.opp_vels = np.empty((0, 2), dtype=np.float64)

        self._publish_obstacle_markers(tracked)

    def _extract_obstacles_from_scan(self, msg: LaserScan) -> np.ndarray:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)

        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment

        # 1. 거리 및 각도 필터링: 전방 [-1.9 rad, +1.9 rad], 유효 거리 [0.35m, 5.0m] (원거리 조기 회피)
        valid_mask = (
            np.isfinite(ranges) &
            (ranges >= max(float(msg.range_min), 0.35)) &
            (ranges <= min(float(msg.range_max), 5.0)) &
            (np.abs(angles) <= 1.9)
        )
        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float64)

        v_ranges = ranges[valid_mask]
        v_angles = angles[valid_mask]

        # 2. base_link 좌표 변환 (라이다 장착 위치 x=0.27m 전방)
        xb = v_ranges * np.cos(v_angles) + 0.27
        yb = v_ranges * np.sin(v_angles)

        # 차량 후방 제외
        fwd_mask = xb > 0.1
        if not np.any(fwd_mask):
            return np.empty((0, 3), dtype=np.float64)

        xb = xb[fwd_mask]
        yb = yb[fwd_mask]

        # 3. map 좌표계 변환
        cos_t = math.cos(self.pose_theta)
        sin_t = math.sin(self.pose_theta)
        xm = self.pose_x + (cos_t * xb - sin_t * yb)
        ym = self.pose_y + (sin_t * xb + cos_t * yb)

        # 4. 정적 맵 벽면 필터링 (Static Map Wall Filter)
        # 벽면과 0.22m 이내인 점은 맵의 정적 벽면으로 간주하여 제거 (센서 노이즈/위치오차 마진 확보)
        dt = self.planner.dt
        orig_x, orig_y, orig_c, orig_s, height, width, resolution = self.planner.map_metainfo

        obs_x = []
        obs_y = []
        for x, y in zip(xm, ym):
            r, c = xy_2_rc(float(x), float(y), orig_x, orig_y, orig_c, orig_s, height, width, resolution)
            if 0 <= r < height and 0 <= c < width:
                if dt[r, c] > 0.22:
                    obs_x.append(float(x))
                    obs_y.append(float(y))

        if len(obs_x) < 6:
            return np.empty((0, 3), dtype=np.float64)

        # 5. 유클리디안 거리 기반 군집화 (Clustering)
        clusters = []
        curr_cluster = [(obs_x[0], obs_y[0])]
        for i in range(1, len(obs_x)):
            dist = math.hypot(obs_x[i] - obs_x[i - 1], obs_y[i] - obs_y[i - 1])
            if dist < 0.28:
                curr_cluster.append((obs_x[i], obs_y[i]))
            else:
                if len(curr_cluster) >= 6:
                    clusters.append(np.array(curr_cluster, dtype=np.float64))
                curr_cluster = [(obs_x[i], obs_y[i])]
        if len(curr_cluster) >= 6:
            clusters.append(np.array(curr_cluster, dtype=np.float64))

        if not clusters:
            return np.empty((0, 3), dtype=np.float64)

        # 6. 클러스터별 중심점(Centroid) 및 헤딩 추정 + 주행 통로(Corridor) 필터
        detected = []
        for cluster in clusters:
            span = np.max(cluster, axis=0) - np.min(cluster, axis=0)
            if math.hypot(span[0], span[1]) > 1.0:
                # 1.0m 초과의 너무 큰 군집은 벽면 오검출일 수 있으므로 제외
                continue

            cx = float(np.mean(cluster[:, 0]))
            cy = float(np.mean(cluster[:, 1]))

            # 가장 가까운 웨이포인트(주행 중심선)와의 거리 검사 (주행 통로 외곽 필터)
            _, nearest_dist, _, nearest_wp_idx = nearest_point(np.array([cx, cy]), self.planner.waypoints[:, :2])
            
            # 주행 레이스라인 기준 좌우 0.40m 이내에 있는 장애물만 실제 회피 대상으로 인정
            # (벽면에 붙은 외곽 허위 장애물 원천 차단)
            if nearest_dist > 0.40:
                continue

            theta = float(self.planner.waypoints[nearest_wp_idx, 3])
            detected.append([cx, cy, theta])

        if not detected:
            return np.empty((0, 3), dtype=np.float64)

        # 차량과의 거리가 가까운 순으로 최대 5개 전달
        detected = np.array(detected, dtype=np.float64)
        dists = np.hypot(detected[:, 0] - self.pose_x, detected[:, 1] - self.pose_y)
        sorted_idx = np.argsort(dists)
        return detected[sorted_idx[:5]]

    def _publish_obstacle_markers(self, tracked_obstacles: list):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        del_marker = Marker()
        del_marker.header.frame_id = 'map'
        del_marker.action = Marker.DELETEALL
        marker_array.markers.append(del_marker)

        marker_id = 1
        for obs in tracked_obstacles:
            # 1. 위치 표시 실린더
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'detected_obstacles'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(obs['x'])
            m.pose.position.y = float(obs['y'])
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = 0.40
            m.scale.y = 0.40
            m.scale.z = 0.30
            # 움직이는 물체는 밝은 주황/빨강, 정지 물체는 보라색
            if obs['speed'] > 0.2:
                m.color = ColorRGBA(r=1.0, g=0.2, b=0.1, a=0.85)
            else:
                m.color = ColorRGBA(r=0.8, g=0.2, b=0.8, a=0.85)
            marker_array.markers.append(m)

            # 2. 속도 벡터 화살표 (속도 >= 0.15 m/s 일 때)
            if obs['speed'] >= 0.15:
                arrow = Marker()
                arrow.header.frame_id = 'map'
                arrow.header.stamp = stamp
                arrow.ns = 'obstacle_velocities'
                arrow.id = marker_id
                marker_id += 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                p_start = Point(x=float(obs['x']), y=float(obs['y']), z=0.15)
                # 속도 벡터 방향으로 0.8초 전방 위치까지 화살표 연장
                p_end = Point(
                    x=float(obs['x'] + obs['vx'] * 0.8),
                    y=float(obs['y'] + obs['vy'] * 0.8),
                    z=0.15
                )
                arrow.points = [p_start, p_end]
                arrow.scale.x = 0.08  # 축 두께
                arrow.scale.y = 0.16  # 머리 두께
                arrow.scale.z = 0.16  # 머리 길이
                arrow.color = ColorRGBA(r=1.0, g=0.9, b=0.0, a=0.95)
                marker_array.markers.append(arrow)

            # 3. 속도 및 ID 텍스트 표시
            txt = Marker()
            txt.header.frame_id = 'map'
            txt.header.stamp = stamp
            txt.ns = 'obstacle_text'
            txt.id = marker_id
            marker_id += 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(obs['x'])
            txt.pose.position.y = float(obs['y'])
            txt.pose.position.z = 0.45
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.25
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            txt.text = f"ID {obs['id']}: {obs['speed']:.2f} m/s"
            marker_array.markers.append(txt)

        self.obstacle_marker_pub.publish(marker_array)

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.pose_x = x
        self.pose_y = y
        self.pose_theta = theta
        self.initial_map_x = x
        self.initial_map_y = y
        self.initial_map_yaw = theta
        self.odom_base_x = None
        self.amcl_received = True
        self.get_logger().info(
            f'Received Initial Pose from RViz2: x={x:.2f}, y={y:.2f}, yaw={theta:.2f}',
            throttle_duration_sec=2.0
        )

    def _initialpose_callback(self, msg: PoseWithCovarianceStamped):
        self.initial_map_x = msg.pose.pose.position.x
        self.initial_map_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.initial_map_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        # RViz2에서 클릭한 시점의 odom 값을 기준점으로 고정
        self.odom_base_x = self.last_raw_odom_x
        self.odom_base_y = self.last_raw_odom_y
        self.odom_base_yaw = self.last_raw_odom_yaw

        # SE(2) 변환: T_map_odom = T_map_base * (T_odom_base)^(-1)
        rel_yaw = self.initial_map_yaw - self.odom_base_yaw
        c = math.cos(rel_yaw)
        s = math.sin(rel_yaw)
        tx = self.initial_map_x - (c * self.odom_base_x - s * self.odom_base_y)
        ty = self.initial_map_y - (s * self.odom_base_x + c * self.odom_base_y)
        self.map_to_odom_transform = (tx, ty, rel_yaw)

        self.pose_x = self.initial_map_x
        self.pose_y = self.initial_map_y
        self.pose_theta = self.initial_map_yaw
        self.amcl_received = True

        self.get_logger().info(
            f'Odom mode Initial Pose set: map=({self.initial_map_x:.2f}, {self.initial_map_y:.2f}, {math.degrees(self.initial_map_yaw):.1f} deg), '
            f'odom_base=({self.odom_base_x:.2f}, {self.odom_base_y:.2f}, {math.degrees(self.odom_base_yaw):.1f} deg)'
        )

    def _odom_velocity_callback(self, msg: Odometry):
        v = msg.twist.twist.linear.x
        self.velocity = v
        self.odom_received = True
        self.get_logger().info(
            f'Received Odom velocity: {v:.2f} m/s',
            throttle_duration_sec=5.0
        )

    def _amcl_pose_fusion_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.pose_x = x
        self.pose_y = y
        self.pose_theta = theta
        self.amcl_received = True
        self.last_odom_x = None
        self.last_odom_y = None
        self.last_odom_yaw = None

    def _odom_fusion_callback(self, msg: Odometry):
        v = msg.twist.twist.linear.x
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        oyaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        self.velocity = v
        self.odom_received = True

        if self.last_odom_x is not None:
            dx_odom = ox - self.last_odom_x
            dy_odom = oy - self.last_odom_y
            dyaw = oyaw - self.last_odom_yaw

            c = math.cos(self.pose_theta)
            s = math.sin(self.pose_theta)
            self.pose_x += c * dx_odom - s * dy_odom
            self.pose_y += s * dx_odom + c * dy_odom
            self.pose_theta += dyaw
            self.pose_theta = math.atan2(math.sin(self.pose_theta), math.cos(self.pose_theta))

        self.last_odom_x = ox
        self.last_odom_y = oy
        self.last_odom_yaw = oyaw

    def _odom_full_callback(self, msg: Odometry):
        self.velocity = msg.twist.twist.linear.x
        self.odom_received = True

        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        oyaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        self.last_raw_odom_x = ox
        self.last_raw_odom_y = oy
        self.last_raw_odom_yaw = oyaw

        if not self.amcl_received or self.odom_base_x is None:
            return

        rel_yaw = self.initial_map_yaw - self.odom_base_yaw
        dx = ox - self.odom_base_x
        dy = oy - self.odom_base_y
        c = math.cos(rel_yaw)
        s = math.sin(rel_yaw)

        self.pose_x = self.initial_map_x + (c * dx - s * dy)
        self.pose_y = self.initial_map_y + (s * dx + c * dy)
        cur_yaw = self.initial_map_yaw + (oyaw - self.odom_base_yaw)
        self.pose_theta = math.atan2(math.sin(cur_yaw), math.cos(cur_yaw))

    def _plan_callback(self):
        if not self.amcl_received:
            self.get_logger().info(
                f'Waiting for Initial 2D Pose Estimate from RViz2... (initialpose received={self.amcl_received}, odom received={self.odom_received})',
                throttle_duration_sec=3.0
            )
            return

        now_time = self.get_clock().now()
        dt = None
        if self._last_plan_time is not None:
            dt = (now_time - self._last_plan_time).nanoseconds * 1e-9
        self._last_plan_time = now_time

        try:
            best_traj, best_cost, traj_cost, abs_v_cost, collision_cost = self.planner.plan(
                self.pose_x, self.pose_y, self.pose_theta,
                self.opp_pose, self.velocity,
                dt=dt, opp_vels=self.opp_vels
            )
        except Exception as e:
            self.get_logger().warn(f'Lattice plan failed: {e}', throttle_duration_sec=2.0)
            self._on_plan_failure()
            return

        if best_traj is None:
            self._on_plan_failure()
        else:
            self.plan_fail_count = 0
            self.best_traj = best_traj
            now = now_time.to_msg()
            self._publish_trajectory(best_traj)
            self._publish_best_traj(best_traj, now)

            if len(self.opp_pose) > 0:
                max_v = float(np.max(np.hypot(self.opp_vels[:, 0], self.opp_vels[:, 1]))) if len(self.opp_vels) > 0 else 0.0
                self.get_logger().info(
                    f'Plan generated [AVOIDANCE ACTIVE]: pose=({self.pose_x:.2f}, {self.pose_y:.2f}), '
                    f'obstacles={len(self.opp_pose)}, max_obs_v={max_v:.2f}m/s, cost={best_cost:.2f}, col_cost={collision_cost:.2f}',
                    throttle_duration_sec=1.5
                )
            else:
                self.get_logger().info(
                    f'Plan generated: pose=({self.pose_x:.2f}, {self.pose_y:.2f}), points={len(best_traj)}, cost={best_cost:.2f}',
                    throttle_duration_sec=2.0
                )

    def _on_plan_failure(self):
        self.plan_fail_count += 1
        if self.plan_fail_count == self.plan_fail_stop_threshold:
            self.get_logger().error(
                f'Lattice plan failed {self.plan_fail_count} times in a row; '
                f'publishing empty trajectory so the controller stops'
            )
        if self.plan_fail_count >= self.plan_fail_stop_threshold:
            self.best_traj = None
            self.trajectory_pub.publish(Float64MultiArray())

    def _publish_trajectory(self, best_traj):
        trajectory = np.ascontiguousarray(best_traj[:, :3], dtype=np.float64)
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label='points',
                size=trajectory.shape[0],
                stride=trajectory.size,
            ),
            MultiArrayDimension(label='xyv', size=3, stride=3),
        ]
        msg.data = trajectory.reshape(-1).tolist()
        self.trajectory_pub.publish(msg)

    def _publish_raceline_vis(self):
        waypoints = self.planner.waypoints
        marker_array = MarkerArray()

        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'raceline'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05
        # 소프트 시안 / 파란색 (눈 피로 감소)
        line.color = ColorRGBA(r=0.0, g=0.6, b=1.0, a=0.6)
        line.pose.orientation.w = 1.0

        for wp in waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.0
            line.points.append(p)
        p = Point()
        p.x = float(waypoints[0, 0])
        p.y = float(waypoints[0, 1])
        p.z = 0.0
        line.points.append(p)

        marker_array.markers.append(line)
        self.raceline_pub.publish(marker_array)

    def _publish_best_traj(self, best_traj, stamp):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'best_traj'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        # 소프트 엘로우 / 골드
        marker.color = ColorRGBA(r=1.0, g=0.85, b=0.1, a=0.95)
        marker.pose.orientation.w = 1.0

        for pt in best_traj:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.0
            marker.points.append(p)

        self.best_traj_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SlamLatticePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()