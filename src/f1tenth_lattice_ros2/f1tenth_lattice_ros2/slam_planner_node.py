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

from f1tenth_lattice_ros2.planner_utils import load_config
from f1tenth_lattice_ros2.lattice_planner import LatticePlanner


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

        self.opp_pose = np.empty((0, 3))
        self.best_traj = None

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
                PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_callback, qos
            )
            self.create_subscription(
                PoseWithCovarianceStamped, 'initialpose', self._amcl_pose_callback, qos
            )
            self.create_subscription(
                Odometry, 'odom', self._odom_full_callback, qos
            )

        self.create_subscription(
            LaserScan, 'scan', self._scan_callback, qos_profile_sensor_data
        )

        # Planned trajectory publisher (read by pure_pursuit_controller_node)
        self.trajectory_pub = self.create_publisher(
            Float64MultiArray, 'planned_trajectory', QoSProfile(depth=1)
        )
        self.raceline_pub = self.create_publisher(MarkerArray, 'raceline_marker', marker_qos)
        self.best_traj_pub = self.create_publisher(Marker, 'best_traj_marker', qos)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Planning Timer
        period = 1.0 / max(self.plan_freq, 1.0)
        self.timer = self.create_timer(period, self._plan_callback)

        # TF Broadcast Timer (50Hz)
        self.tf_timer = self.create_timer(0.02, self._publish_tf_timer)

        # Raceline visualization marker (1s interval)
        self.create_timer(1.0, self._publish_raceline_vis)

        self.get_logger().info(
            f'SLAM lattice planner ready (mode={self.localization_mode}, plan_freq={self.plan_freq:.1f}Hz, '
            f'max_speed={self.max_speed}m/s, max_steer={self.max_steer:.3f}rad, trajectory isolated)'
        )

    def _publish_tf_timer(self):
        if not self.amcl_received:
            return

        ox_base = self.odom_base_x if self.odom_base_x is not None else 0.0
        oy_base = self.odom_base_y if self.odom_base_y is not None else 0.0
        oyaw_base = self.odom_base_yaw if self.odom_base_yaw is not None else 0.0

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        map_odom_yaw = self.initial_map_yaw - oyaw_base
        c = math.cos(map_odom_yaw)
        s = math.sin(map_odom_yaw)
        t.transform.translation.x = float(self.initial_map_x - (c * ox_base - s * oy_base))
        t.transform.translation.y = float(self.initial_map_y - (s * ox_base + c * oy_base))
        t.transform.translation.z = 0.0

        cy = math.cos(map_odom_yaw * 0.5)
        sy = math.sin(map_odom_yaw * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(sy)
        t.transform.rotation.w = float(cy)
        self.tf_broadcaster.sendTransform(t)

    def _scan_callback(self, msg: LaserScan):
        pass

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
        v = msg.twist.twist.linear.x
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        oyaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        self.velocity = v
        self.odom_received = True

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

    def _plan_callback(self):
        if not self.amcl_received:
            self.get_logger().info(
                f'Waiting for Initial 2D Pose Estimate from RViz2... (initialpose received={self.amcl_received}, odom received={self.odom_received})',
                throttle_duration_sec=3.0
            )
            return

        try:
            best_traj, best_cost, traj_cost, abs_v_cost, collision_cost = self.planner.plan(
                self.pose_x, self.pose_y, self.pose_theta,
                self.opp_pose, self.velocity
            )
        except Exception as e:
            self.get_logger().warn(f'Lattice plan failed: {e}', throttle_duration_sec=2.0)
            return

        if best_traj is not None:
            self.best_traj = best_traj
            now = self.get_clock().now().to_msg()
            self._publish_trajectory(best_traj)
            self._publish_best_traj(best_traj, now)

            self.get_logger().info(
                f'Plan generated: pose=({self.pose_x:.2f}, {self.pose_y:.2f}), points={len(best_traj)}, cost={best_cost:.2f}',
                throttle_duration_sec=2.0
            )

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