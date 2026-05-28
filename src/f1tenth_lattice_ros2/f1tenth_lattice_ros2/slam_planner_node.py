import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

import numpy as np
import math

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_lattice_ros2.planner_utils import load_config
from f1tenth_lattice_ros2.lattice_planner import LatticePlanner


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SlamLatticePlannerNode(Node):
    def __init__(self):
        super().__init__('slam_lattice_planner')

        self.declare_parameter('config_path', '')
        self.declare_parameter('raceline_path', '')
        self.declare_parameter('map_path', '')
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('plan_frequency', 10.0)
        
        self.declare_parameter('localization_mode', 'fusion')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('odom_cov_scale', 100.0)

        config_path = self.get_parameter('config_path').value
        raceline_path = self.get_parameter('raceline_path').value
        map_path = self.get_parameter('map_path').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steer = self.get_parameter('max_steering_angle').value
        plan_freq = self.get_parameter('plan_frequency').value
        opponent_ns = ''  # No opponent namespace for single-vehicle operation
        self.localization_mode = self.get_parameter('localization_mode').value
        self.initial_x = self.get_parameter('initial_x').value
        self.initial_y = self.get_parameter('initial_y').value
        self.initial_yaw = self.get_parameter('initial_yaw').value
        self.odom_cov_scale = self.get_parameter('odom_cov_scale').value

        if not config_path or not raceline_path or not map_path:
            self.get_logger().fatal('config_path, raceline_path, map_path parameters are required')
            raise RuntimeError('Missing required parameters')

        self.get_logger().info(f'Loading config from: {config_path}')
        self.get_logger().info(f'Loading raceline from: {raceline_path}')
        self.get_logger().info(f'Loading map from: {map_path}')

        ns = self.get_namespace().strip('/')
        conf = load_config(config_path, namespace=ns)
        self.planner = LatticePlanner(conf, map_path, raceline_path)
        self.get_logger().info('Lattice planner initialized')

        self.pose_x = self.initial_x
        self.pose_y = self.initial_y
        self.pose_theta = self.initial_yaw
        self.velocity = 0.0
        self.odom_received = False
        self.latest_odom_stamp = None

        self.amcl_x = self.initial_x
        self.amcl_y = self.initial_y
        self.amcl_theta = self.initial_yaw
        self.amcl_cov_xx = 1.0
        self.amcl_cov_yy = 1.0
        self.amcl_cov_aa = 1.0
        self.amcl_received = False
        self.amcl_stamp = None

        self.odom_x = self.initial_x
        self.odom_y = self.initial_y
        self.odom_theta = self.initial_yaw
        self.odom_cov_xx = self.odom_cov_scale
        self.odom_cov_yy = self.odom_cov_scale
        self.odom_cov_aa = self.odom_cov_scale
        self.odom_stamp = None

        self.opp_pose = np.empty((0, 3))  # No opponent tracking in current real-vehicle setup

        qos = QoSProfile(depth=10)

        if self.localization_mode == 'fusion':
            self.get_logger().info('Localization mode: FUSION (AMCL + Odometry)')
            self.create_subscription(
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_callback, qos
            )
            self.odom_sub = self.create_subscription(
                Odometry, 'odom', self._odom_callback, qos
            )
        elif self.localization_mode == 'amcl':
            self.get_logger().info('Localization mode: AMCL ONLY')
            self.create_subscription(
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_only_callback, qos
            )
            self.odom_sub = self.create_subscription(
                Odometry, 'odom', self._odom_velocity_only_callback, qos
            )
        else:
            self.get_logger().info('Localization mode: ODOMETRY ONLY')
            self.odom_sub = self.create_subscription(
                Odometry, 'odom', self._odom_callback, qos
            )

        # Opponent/head-to-head support removed for single-vehicle operation

        self.drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', qos)
        self.raceline_pub = self.create_publisher(MarkerArray, 'raceline_marker', qos)
        self.best_traj_pub = self.create_publisher(Marker, 'best_traj_marker', qos)

        period = 1.0 / plan_freq
        self.timer = self.create_timer(period, self._plan_callback)

        self.create_timer(2.0, self._publish_raceline_once)
        self._raceline_published = False

        self.get_logger().info(
            f'SLAM lattice planner ready (mode={self.localization_mode}, plan_freq={plan_freq:.1f}Hz, '
            f'max_speed={self.max_speed}m/s, max_steer={self.max_steer:.3f}rad)'
        )

    def _odom_callback(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.odom_theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.velocity = msg.twist.twist.linear.x

        cov = msg.pose.covariance
        self.odom_cov_xx = max(cov[0], 0.01)
        self.odom_cov_yy = max(cov[7], 0.01)
        self.odom_cov_aa = max(cov[35], 0.01)

        self.latest_odom_stamp = msg.header.stamp
        self.odom_stamp = msg.header.stamp
        self.odom_received = True

        if self.localization_mode == 'odom':
            self.pose_x = self.odom_x
            self.pose_y = self.odom_y
            self.pose_theta = self.odom_theta
        elif self.localization_mode == 'fusion':
            if self.amcl_received:
                self._fuse_poses()
            else:
                self.pose_x = self.odom_x
                self.pose_y = self.odom_y
                self.pose_theta = self.odom_theta

        if getattr(self, 'best_traj', None) is not None:
            steering, speed = self.planner.tracker.plan(
                self.pose_x, self.pose_y, self.pose_theta,
                self.velocity, self.best_traj
            )
            steering = float(np.clip(steering, -self.max_steer, self.max_steer))
            speed = float(np.clip(speed, 0.0, self.max_speed))

            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = msg.header.stamp
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = steering
            self.drive_pub.publish(drive_msg)

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.amcl_x = msg.pose.pose.position.x
        self.amcl_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.amcl_theta = quat_to_yaw(q.x, q.y, q.z, q.w)

        cov = msg.pose.covariance
        self.amcl_cov_xx = max(cov[0], 0.001)
        self.amcl_cov_yy = max(cov[7], 0.001)
        self.amcl_cov_aa = max(cov[35], 0.001)

        self.latest_odom_stamp = msg.header.stamp
        self.amcl_stamp = msg.header.stamp
        self.amcl_received = True

        if self.odom_received:
            self._fuse_poses()

    def _amcl_pose_only_callback(self, msg: PoseWithCovarianceStamped):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose_theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.latest_odom_stamp = msg.header.stamp
        self.odom_received = True

    def _odom_velocity_only_callback(self, msg: Odometry):
        self.velocity = msg.twist.twist.linear.x

    def _fuse_poses(self):
        now = self.get_clock().now()
        amcl_age = 0.0
        odom_age = 0.0
        if self.amcl_stamp is not None:
            amcl_age = (now - self.amcl_stamp).nanoseconds * 1e-9
        if self.odom_stamp is not None:
            odom_age = (now - self.odom_stamp).nanoseconds * 1e-9

        amcl_decay = 1.0
        if amcl_age > 1.0:
            amcl_decay = max(0.2, 1.0 - 0.25 * (amcl_age - 1.0))

        odom_decay = 1.0
        if odom_age > 0.5:
            odom_decay = max(0.3, 1.0 - 0.5 * (odom_age - 0.5))

        amcl_trust_x = amcl_decay / max(self.amcl_cov_xx, 1e-4)
        amcl_trust_y = amcl_decay / max(self.amcl_cov_yy, 1e-4)
        odom_trust_x = odom_decay / max(self.odom_cov_xx / self.odom_cov_scale, 1e-4)
        odom_trust_y = odom_decay / max(self.odom_cov_yy / self.odom_cov_scale, 1e-4)

        total_trust_x = amcl_trust_x + odom_trust_x
        total_trust_y = amcl_trust_y + odom_trust_y
        amcl_weight_x = amcl_trust_x / total_trust_x
        odom_weight_x = odom_trust_x / total_trust_x
        amcl_weight_y = amcl_trust_y / total_trust_y
        odom_weight_y = odom_trust_y / total_trust_y

        amcl_trust_a = amcl_decay / max(self.amcl_cov_aa, 1e-4)
        odom_trust_a = odom_decay / max(self.odom_cov_aa / self.odom_cov_scale, 1e-4)
        total_trust_a = amcl_trust_a + odom_trust_a
        amcl_weight_a = amcl_trust_a / total_trust_a

        self.pose_x = amcl_weight_x * self.amcl_x + odom_weight_x * self.odom_x
        self.pose_y = amcl_weight_y * self.amcl_y + odom_weight_y * self.odom_y
        self.pose_theta = self._circular_mean(self.amcl_theta, self.odom_theta, amcl_weight_a)

    def _circular_mean(self, angle1, angle2, weight1):
        x1 = np.cos(angle1)
        y1 = np.sin(angle1)
        x2 = np.cos(angle2)
        y2 = np.sin(angle2)
        x_mean = weight1 * x1 + (1 - weight1) * x2
        y_mean = weight1 * y1 + (1 - weight1) * y2
        return math.atan2(y_mean, x_mean)

    def _opp_odom_callback(self, msg: Odometry):
        # Opponent odometry callback removed for single-vehicle operation
        pass

    def _plan_callback(self):
        if not self.odom_received:
            return

        try:
            best_traj, best_cost, traj_cost, abs_v_cost, collision_cost = self.planner.plan(
                self.pose_x, self.pose_y, self.pose_theta,
                np.empty((0, 3)), self.velocity
            )
            self.best_traj = best_traj
        except Exception as e:
            self.get_logger().warn(f'Lattice plan failed: {e}', throttle_duration_sec=2.0)
            return

        now = self.get_clock().now().to_msg()

        self._publish_best_traj(self.best_traj, now)

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
