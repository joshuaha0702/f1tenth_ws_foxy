import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64MultiArray, String

from f1tenth_lattice_ros2.planner_utils import load_config
from f1tenth_lattice_ros2.pure_pursuit import PurePursuitPlanner


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuitControllerNode(Node):
    """Fast odom-to-drive controller isolated from CPU-heavy lattice planning."""

    def __init__(self):
        super().__init__('pure_pursuit_controller')

        self.declare_parameter('config_path', '')
        self.declare_parameter('raceline_path', '')
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('max_steering_angle', 0.4189)

        config_path = self.get_parameter('config_path').value
        raceline_path = self.get_parameter('raceline_path').value
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer = float(self.get_parameter('max_steering_angle').value)

        if not config_path or not raceline_path:
            self.get_logger().fatal('config_path and raceline_path are required')
            raise RuntimeError('Missing required parameters')

        namespace = self.get_namespace().strip('/')
        conf = load_config(config_path, namespace=namespace)
        self.tracker = PurePursuitPlanner(conf, raceline_path)

        # Compile the numba Pure Pursuit path before odom messages start arriving.
        dummy_traj = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        self.tracker.plan(0.0, 0.0, 0.0, 0.0, dummy_traj)
        self.tracker.prev_error = 0.0

        self.best_traj = None
        self.publishing_enabled = True
        self.stop_pending = False
        self.drive_count = 0
        self.first_drive_stamp_ns = None

        odom_qos = QoSProfile(depth=20)
        qos = QoSProfile(depth=10)
        trajectory_qos = QoSProfile(depth=1)

        self.create_subscription(
            Odometry, 'odom', self._odom_callback, odom_qos
        )
        self.create_subscription(
            Float64MultiArray,
            'planned_trajectory',
            self._trajectory_callback,
            trajectory_qos,
        )
        self.create_subscription(
            String, '/episode/control', self._episode_control_callback, qos
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, 'drive', qos
        )

        self.get_logger().info(
            'Pure Pursuit controller ready (separate process, odom-triggered)'
        )

    def _trajectory_callback(self, msg: Float64MultiArray):
        if not msg.data:
            self.best_traj = None
            return
        if len(msg.data) % 3 != 0:
            self.get_logger().warn(
                f'Invalid trajectory payload length: {len(msg.data)}',
                throttle_duration_sec=2.0,
            )
            return
        trajectory = np.asarray(msg.data, dtype=np.float64).reshape((-1, 3))
        if trajectory.shape[0] < 2:
            return
        self.best_traj = trajectory

    def _odom_callback(self, msg: Odometry):
        # STOP 직후의 첫 odom 시각에 정지 명령을 맞춰 발행한다. 콜백에서
        # get_clock().now()를 사용하면 이미 처리된 최신 odom보다 과거 시각이
        # 기록될 수 있어 episode bag의 drive header가 역행할 수 있다.
        if self.stop_pending:
            self.stop_pending = False
            stop_msg = AckermannDriveStamped()
            stop_msg.header.stamp = msg.header.stamp
            stop_msg.header.frame_id = 'base_link'
            stop_msg.drive.speed = 0.0
            stop_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(stop_msg)
            return

        trajectory = self.best_traj
        if not self.publishing_enabled or trajectory is None:
            return

        q = msg.pose.pose.orientation
        pose_theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        steering, speed = self.tracker.plan(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            pose_theta,
            msg.twist.twist.linear.x,
            trajectory,
        )

        drive_msg = AckermannDriveStamped()
        # Preserve the triggering odom timestamp so sim-time control rate is
        # directly measurable from recorded drive messages.
        drive_msg.header.stamp = msg.header.stamp
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = float(np.clip(speed, 0.0, self.max_speed))
        drive_msg.drive.steering_angle = float(
            np.clip(steering, -self.max_steer, self.max_steer)
        )
        self.drive_pub.publish(drive_msg)
        self._update_rate_diagnostics(msg)

    def _update_rate_diagnostics(self, odom_msg: Odometry):
        stamp = odom_msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if self.first_drive_stamp_ns is None:
            self.first_drive_stamp_ns = stamp_ns
            self.drive_count = 1
            return

        self.drive_count += 1
        if self.drive_count % 500 != 0:
            return

        elapsed = (stamp_ns - self.first_drive_stamp_ns) * 1e-9
        if elapsed > 0.0:
            rate = (self.drive_count - 1) / elapsed
            self.get_logger().info(
                f'[control rate] {rate:.2f} Hz sim-time '
                f'({self.drive_count} drive messages)'
            )

    def _episode_control_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command == 'STOP':
            was_enabled = self.publishing_enabled
            self.publishing_enabled = False
            self.best_traj = None
            self.drive_count = 0
            self.first_drive_stamp_ns = None
            # EpisodeManager가 같은 명령을 재발행해도 정지 명령은 한 번만 보낸다.
            if was_enabled:
                self.stop_pending = True
                self.get_logger().info(
                    '[episode] STOP — controller paused; '
                    'zero command pending next odom'
                )
        elif command == 'START':
            if self.publishing_enabled:
                return
            self.publishing_enabled = True
            self.drive_count = 0
            self.first_drive_stamp_ns = None
            self.get_logger().info('[episode] START — controller resumed')
        else:
            self.get_logger().warn(
                f'[episode] unknown control cmd: {msg.data!r}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
