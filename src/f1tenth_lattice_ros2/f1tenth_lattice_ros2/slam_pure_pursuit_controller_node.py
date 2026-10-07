#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64MultiArray, String

from f1tenth_lattice_ros2.planner_utils import load_config
from f1tenth_lattice_ros2.pure_pursuit import PurePursuitPlanner


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SlamPurePursuitControllerNode(Node):
    """
    100Hz Isolated Pure Pursuit Controller for Real-Car / SLAM Environment.
    Runs on an independent process and 100Hz timer, isolated from CPU-heavy Lattice Planner.
    """

    def __init__(self):
        super().__init__('slam_pure_pursuit_controller')

        # --- Parameters ---
        self.declare_parameter('config_path', '')
        self.declare_parameter('raceline_path', '')
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('max_steering_angle', 0.26)
        # 조향 변화율 제한 [rad/s]. 틱당 한계는 control_frequency로 환산한다.
        # (플래너 통합 시절 0.08 rad / 10 Hz plan = 0.8 rad/s 와 동일한 실효 제한)
        self.declare_parameter('max_steer_rate', 0.8)
        self.declare_parameter('control_frequency', 100.0)
        # 워치독: 입력이 이 시간 이상 끊기면 0-cmd로 전환 (플래너 사망/odom 두절 시 개루프 주행 방지)
        self.declare_parameter('trajectory_timeout', 0.5)   # 10 Hz 계획 기준 5 주기
        self.declare_parameter('odom_timeout', 0.2)
        self.declare_parameter('localization_mode', 'scan')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)

        config_path = self.get_parameter('config_path').value
        raceline_path = self.get_parameter('raceline_path').value
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer = float(self.get_parameter('max_steering_angle').value)
        self.control_freq = max(float(self.get_parameter('control_frequency').value), 1.0)
        self.max_steer_rate = float(self.get_parameter('max_steer_rate').value)
        self.max_steer_step = self.max_steer_rate / self.control_freq   # 틱당 허용 조향 변화 [rad]
        self.trajectory_timeout = float(self.get_parameter('trajectory_timeout').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.loc_mode = str(self.get_parameter('localization_mode').value).lower()
        self.initial_x = float(self.get_parameter('initial_x').value)
        self.initial_y = float(self.get_parameter('initial_y').value)
        self.initial_yaw = float(self.get_parameter('initial_yaw').value)

        if not config_path or not raceline_path:
            self.get_logger().fatal('config_path and raceline_path are required parameters')
            raise RuntimeError('Missing required parameters')

        ns = self.get_namespace().strip('/')
        if not ns:
            ns = 'car1'

        conf = load_config(config_path, namespace=ns)
        self.tracker = PurePursuitPlanner(conf, raceline_path)

        # Warm-up Numba JIT path before control timer begins
        dummy_traj = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
        self.tracker.plan(0.0, 0.0, 0.0, 0.0, dummy_traj)
        self.tracker.prev_error = 0.0

        # --- State Variables ---
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
        self.last_raw_odom_x = 0.0
        self.last_raw_odom_y = 0.0
        self.last_raw_odom_yaw = 0.0

        self.last_steer = 0.0
        self.best_traj = None
        self.publishing_enabled = True
        self.last_traj_time = None   # rclpy.time.Time, 마지막 planned_trajectory 수신
        self.last_odom_time = None   # rclpy.time.Time, 마지막 odom 수신
        self.stale_reason = None     # 워치독이 멈춘 이유 (복구 로그용)

        # Diagnostics & Timing
        self.drive_count = 0
        self.first_drive_stamp_ns = None
        self.last_control_time = None

        # --- QoS Profiles ---
        qos = QoSProfile(depth=10)
        odom_qos = QoSProfile(depth=20)
        trajectory_qos = QoSProfile(depth=1)

        # --- Subscriptions ---
        # 1. Planned trajectory from planner node
        self.create_subscription(
            Float64MultiArray, 'planned_trajectory', self._trajectory_callback, trajectory_qos
        )

        # 2. Episode control (optional)
        self.create_subscription(
            String, '/episode/control', self._episode_control_callback, qos
        )

        # 3. Localization & Odometry
        if self.loc_mode in ['scan', 'amcl']:
            self.create_subscription(
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_callback, qos
            )
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._amcl_pose_callback, qos
            )
        elif self.loc_mode == 'fusion':
            self.create_subscription(
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_fusion_callback, qos
            )
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._amcl_pose_fusion_callback, qos
            )
        else:  # odom mode
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._initialpose_callback, qos
            )

        # Odometry state update subscription
        self.create_subscription(
            Odometry, 'odom', self._odom_callback, odom_qos
        )

        # --- Publisher ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', qos)

        # --- 100Hz Control Timer ---
        timer_period = 1.0 / self.control_freq
        self.control_timer = self.create_timer(timer_period, self._control_timer_callback)

        self.get_logger().info(
            f'SLAM Pure Pursuit Controller ready (mode={self.loc_mode}, '
            f'control_freq={self.control_freq:.1f}Hz, max_speed={self.max_speed}m/s, '
            f'max_steer={self.max_steer:.3f}rad, max_steer_rate={self.max_steer_rate:.2f}rad/s '
            f'({self.max_steer_step:.4f}rad/tick), traj_timeout={self.trajectory_timeout}s, '
            f'odom_timeout={self.odom_timeout}s)'
        )

    def _trajectory_callback(self, msg: Float64MultiArray):
        if not msg.data:
            # 플래너가 계획 실패를 알리는 빈 경로: 타임아웃을 기다리지 않고 즉시 정지
            self.best_traj = None
            return
        if len(msg.data) % 3 != 0:
            return
        trajectory = np.asarray(msg.data, dtype=np.float64).reshape((-1, 3))
        if trajectory.shape[0] < 2:
            return
        self.best_traj = trajectory
        self.last_traj_time = self.get_clock().now()

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
        self.last_odom_x = None
        self.last_odom_y = None
        self.last_odom_yaw = None
        self.amcl_received = True

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

    def _initialpose_callback(self, msg: PoseWithCovarianceStamped):
        self.initial_map_x = msg.pose.pose.position.x
        self.initial_map_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.initial_map_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        # RViz2에서 클릭한 시점의 odom 값을 기준점으로 고정
        self.odom_base_x = self.last_raw_odom_x
        self.odom_base_y = self.last_raw_odom_y
        self.odom_base_yaw = self.last_raw_odom_yaw

        self.pose_x = self.initial_map_x
        self.pose_y = self.initial_map_y
        self.pose_theta = self.initial_map_yaw
        self.amcl_received = True

        self.get_logger().info(
            f'Controller Odom mode Initial Pose set: map=({self.initial_map_x:.2f}, {self.initial_map_y:.2f}, {math.degrees(self.initial_map_yaw):.1f} deg), '
            f'odom_base=({self.odom_base_x:.2f}, {self.odom_base_y:.2f}, {math.degrees(self.odom_base_yaw):.1f} deg)'
        )

    def _odom_callback(self, msg: Odometry):
        v = msg.twist.twist.linear.x
        self.velocity = v
        self.odom_received = True
        self.last_odom_time = self.get_clock().now()

        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        oyaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        self.last_raw_odom_x = ox
        self.last_raw_odom_y = oy
        self.last_raw_odom_yaw = oyaw

        # Odom Dead-Reckoning update to bridge inter-AMCL updates
        if self.loc_mode in ['scan', 'amcl', 'fusion']:
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
        else:  # odom mode
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

    def _input_stale_reason(self, now):
        """워치독: 경로/odom 중 하나라도 타임아웃이면 이유 문자열, 정상이면 None."""
        if self.last_traj_time is None:
            return 'no trajectory yet'
        traj_age = (now - self.last_traj_time).nanoseconds * 1e-9
        if traj_age > self.trajectory_timeout:
            return f'trajectory stale ({traj_age:.2f}s > {self.trajectory_timeout}s)'
        if self.last_odom_time is None:
            return 'no odom yet'
        odom_age = (now - self.last_odom_time).nanoseconds * 1e-9
        if odom_age > self.odom_timeout:
            return f'odom stale ({odom_age:.2f}s > {self.odom_timeout}s)'
        return None

    def _publish_stop(self, stamp):
        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = stamp
        stop_msg.header.frame_id = 'base_link'
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(stop_msg)
        self.last_steer = 0.0

    def _control_timer_callback(self):
        if not self.publishing_enabled or not self.amcl_received:
            return

        now = self.get_clock().now()

        # 워치독: 플래너 사망·계획 실패·odom 두절 시 낡은 입력으로 개루프 주행하지 않도록
        # 매 틱 0-cmd를 계속 보내 액추에이터를 정지 상태로 유지한다 (한 번만 보내면 VESC가
        # 마지막 명령을 유지할 수 있음).
        reason = self._input_stale_reason(now)
        if reason is None and self.best_traj is None:
            reason = 'trajectory cleared by planner'
        if reason is not None:
            if self.stale_reason != reason:
                self.get_logger().warn(f'[watchdog] stopping: {reason}')
                self.stale_reason = reason
            self._publish_stop(now.to_msg())
            return
        if self.stale_reason is not None:
            self.get_logger().info('[watchdog] inputs healthy again, resuming control')
            self.stale_reason = None

        trajectory = self.best_traj

        # Ultra-fast Pure Pursuit control calculation (< 0.05ms)
        steering, speed = self.tracker.plan(
            self.pose_x, self.pose_y, self.pose_theta,
            self.velocity, trajectory
        )

        # Steering rate limit [rad/s -> rad/tick] & saturation
        steer_diff = steering - self.last_steer
        steer_diff = max(min(steer_diff, self.max_steer_step), -self.max_steer_step)
        steering = self.last_steer + steer_diff
        self.last_steer = steering

        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        speed = float(np.clip(speed, 0.0, self.max_speed))

        # Publish 100Hz Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = now.to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering
        self.drive_pub.publish(drive_msg)

        self._update_rate_diagnostics(now)

    def _update_rate_diagnostics(self, now_time):
        stamp_ns = now_time.nanoseconds
        if self.first_drive_stamp_ns is None:
            self.first_drive_stamp_ns = stamp_ns
            self.drive_count = 1
            return

        self.drive_count += 1
        if self.drive_count % 1000 != 0:
            return

        elapsed = (stamp_ns - self.first_drive_stamp_ns) * 1e-9
        if elapsed > 0.0:
            rate = (self.drive_count - 1) / elapsed
            self.get_logger().info(
                f'[control rate] {rate:.2f} Hz ({self.drive_count} drive messages, '
                f'pose=({self.pose_x:.2f}, {self.pose_y:.2f}), steer={self.last_steer:.3f}rad)'
            )

    def _episode_control_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command == 'STOP':
            self.publishing_enabled = False
            self.best_traj = None
            self.drive_count = 0
            self.first_drive_stamp_ns = None
            self._publish_stop(self.get_clock().now().to_msg())
        elif command == 'START':
            self.publishing_enabled = True
            self.drive_count = 0
            self.first_drive_stamp_ns = None
            self.last_steer = 0.0      # 이전 에피소드의 조향값에서 램프업하지 않도록
            self.stale_reason = None


def main(args=None):
    rclpy.init(args=args)
    node = SlamPurePursuitControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
