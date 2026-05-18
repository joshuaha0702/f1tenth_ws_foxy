import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np
import math

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PoseStamped
# [하드웨어 호환성] 실제 차량 구동을 위한 Ackermann 메시지 임포트
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_lattice_ros2.planner_utils import load_config
from f1tenth_lattice_ros2.lattice_planner import LatticePlanner


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class LatticePlannerNode(Node):
    def __init__(self):
        super().__init__('lattice_planner')

        # Parameters
        self.declare_parameter('config_path', '')
        self.declare_parameter('raceline_path', '')
        self.declare_parameter('map_path', '')
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('plan_frequency', 10.0)
        self.declare_parameter('opponent_namespace', '')

        config_path = self.get_parameter('config_path').value
        raceline_path = self.get_parameter('raceline_path').value
        map_path = self.get_parameter('map_path').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steer = self.get_parameter('max_steering_angle').value
        plan_freq = self.get_parameter('plan_frequency').value
        opponent_ns = self.get_parameter('opponent_namespace').value

        if not config_path or not raceline_path or not map_path:
            self.get_logger().fatal(
                'config_path, raceline_path, map_path parameters are required'
            )
            raise RuntimeError('Missing required parameters')

        # Initialize planner
        self.get_logger().info(f'Loading config from: {config_path}')
        self.get_logger().info(f'Loading raceline from: {raceline_path}')
        self.get_logger().info(f'Loading map from: {map_path}')

        ns = self.get_namespace().strip('/')
        conf = load_config(config_path, namespace=ns)
        self.planner = LatticePlanner(conf, map_path, raceline_path)
        self.get_logger().info('Lattice planner initialized')

        # Vehicle state
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.velocity = 0.0
        self.odom_received = False
        self.latest_odom_stamp = None

        # Opponent state (head-to-head mode)
        self.opp_pose = np.empty((0, 3))

        # QoS
        qos = QoSProfile(depth=10)

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_callback, qos
        )
        if opponent_ns:
            self.create_subscription(
                Odometry, f'/{opponent_ns}/odom', self._opp_odom_callback, qos
            )
            self.get_logger().info(f'Head-to-head mode: tracking opponent /{opponent_ns}/odom')

        # [하드웨어 호환성] 퍼블리셔를 AckermannDriveStamped 타입으로 변경 (실제 차량 VESC 대응)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', qos)
        self.raceline_pub = self.create_publisher(MarkerArray, 'raceline_marker', qos)
        self.best_traj_pub = self.create_publisher(Marker, 'best_traj_marker', qos)

        # Planning timer
        period = 1.0 / plan_freq
        self.timer = self.create_timer(period, self._plan_callback)

        # Publish raceline visualization once after init
        self.create_timer(2.0, self._publish_raceline_once)
        self._raceline_published = False

        self.get_logger().info(
            f'Lattice planner node ready (plan_freq={plan_freq:.1f}Hz, '
            f'max_speed={self.max_speed}m/s, max_steer={self.max_steer:.3f}rad)'
        )

    def _odom_callback(self, msg: Odometry):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose_theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.velocity = msg.twist.twist.linear.x
        self.latest_odom_stamp = msg.header.stamp
        self.odom_received = True

    def _opp_odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.opp_pose = np.array([[x, y, theta]])

    def _plan_callback(self):
        if not self.odom_received:
            return

        opp_poses = self.opp_pose

        try:
            best_traj, best_cost, traj_cost, abs_v_cost, collision_cost = self.planner.plan(
                self.pose_x, self.pose_y, self.pose_theta,
                opp_poses, self.velocity
            )
        except Exception as e:
            self.get_logger().warn(f'Lattice plan failed: {e}', throttle_duration_sec=2.0)
            return

        # Pure pursuit on the selected local trajectory
        steering, speed = self.planner.tracker.plan(
            self.pose_x, self.pose_y, self.pose_theta,
            self.velocity, best_traj
        )

        # Clamp outputs
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        speed = float(np.clip(speed, 0.0, self.max_speed))

        now = self.get_clock().now().to_msg()

        # [하드웨어 호환성] Ackermann 메시지 생성 및 필드 할당 (Twist 대신 직접 필드 사용)
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.latest_odom_stamp if self.latest_odom_stamp else now
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering
        self.drive_pub.publish(drive_msg)

        # Publish best trajectory visualization
        self._publish_best_traj(best_traj, now)

    def _publish_raceline_once(self):
        if self._raceline_published:
            return
        self._raceline_published = True

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
        line.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        line.pose.orientation.w = 1.0

        for wp in waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.0
            line.points.append(p)
        # Close loop
        p = Point()
        p.x = float(waypoints[0, 0])
        p.y = float(waypoints[0, 1])
        p.z = 0.0
        line.points.append(p)

        marker_array.markers.append(line)
        self.raceline_pub.publish(marker_array)
        self.get_logger().info('Raceline visualization published')

    def _publish_best_traj(self, best_traj, stamp):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'best_traj'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
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
    node = LatticePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
