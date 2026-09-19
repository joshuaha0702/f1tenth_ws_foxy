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
        self.declare_parameter('max_steer_rate', 0.08)
        self.declare_parameter('control_frequency', 100.0)
        self.declare_parameter('localization_mode', 'scan')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)

        config_path = self.get_parameter('config_path').value
        raceline_path = self.get_parameter('raceline_path').value
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer = float(self.get_parameter('max_steering_angle').value)
        self.max_steer_rate = float(self.get_parameter('max_steer_rate').value)
        self.control_freq = float(self.get_parameter('control_frequency').value)
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

        self.last_steer = 0.0
        self.best_traj = None
        self.publishing_enabled = True

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
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_callback, qos
            )
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._amcl_pose_callback, qos
            )

        # Odometry state update subscription
        self.create_subscription(
            Odometry, 'odom', self._odom_callback, odom_qos
        )

        # --- Publisher ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', qos)

        # --- 100Hz Control Timer ---
        timer_period = 1.0 / max(self.control_freq, 1.0)
        self.control_timer = self.create_timer(timer_period, self._control_timer_callback)

        self.get_logger().info(
            f'SLAM Pure Pursuit Controller ready (mode={self.loc_mode}, '
            f'control_freq={self.control_freq:.1f}Hz, max_speed={self.max_speed}m/s, '
            f'max_steer={self.max_steer:.3f}rad)'
        )

    def _trajectory_callback(self, msg: Float64MultiArray):
        if not msg.data:
            self.best_traj = None
            return
        if len(msg.data) % 3 != 0:
            return
        trajectory = np.asarray(msg.data, dtype=np.float64).reshape((-1, 3))
        if trajectory.shape[0] < 2:
            return
        self.best_traj = trajectory

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

    def _odom_callback(self, msg: Odometry):
        v = msg.twist.twist.linear.x
        self.velocity = v
        self.odom_received = True

        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        oyaw = quat_to_yaw(q.x, q.y, q.z, q.w)

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
            if not self.amcl_received:
                return
            if self.odom_base_x is None:
                self.odom_base_x = ox
                self.odom_base_y = oy
                self.odom_base_yaw = oyaw

            dx = ox - self.odom_base_x
            dy = oy - self.odom_base_y
            dyaw = oyaw - self.odom_base_yaw

            delta_yaw = self.initial_map_yaw - self.odom_base_yaw
            c = math.cos(delta_yaw)
            s = math.sin(delta_yaw)
            self.pose_x = self.initial_map_x + (c * dx - s * dy)
            self.pose_y = self.initial_map_y + (s * dx + c * dy)
            self.pose_theta = math.atan2(
                math.sin(self.initial_map_yaw + dyaw),
                math.cos(self.initial_map_yaw + dyaw)
            )

    def _control_timer_callback(self):
        trajectory = self.best_traj
        if not self.publishing_enabled or trajectory is None or not self.amcl_received:
            return

        now = self.get_clock().now()

        # Ultra-fast Pure Pursuit control calculation (< 0.05ms)
        steering, speed = self.tracker.plan(
            self.pose_x, self.pose_y, self.pose_theta,
            self.velocity, trajectory
        )

        # Steering rate limit & saturation
        steer_diff = steering - self.last_steer
        steer_diff = max(min(steer_diff, self.max_steer_rate), -self.max_steer_rate)
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
            stop_msg = AckermannDriveStamped()
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link'
            stop_msg.drive.speed = 0.0
            stop_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(stop_msg)
        elif command == 'START':
            self.publishing_enabled = True
            self.drive_count = 0
            self.first_drive_stamp_ns = None


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
