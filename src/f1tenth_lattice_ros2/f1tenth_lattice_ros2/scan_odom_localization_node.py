import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

import numpy as np
import math
import os
import glob
import yaml
import open3d as o3d
from PIL import Image

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
import tf2_ros


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class ScanOdomLocalizationNode(Node):
    def __init__(self):
        super().__init__('scan_odom_localization_node')

        self.declare_parameter('localization_mode', 'fusion')
        self.declare_parameter('map_path', '')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('odom_cov_scale', 50.0)
        self.declare_parameter('icp_max_distance', 0.2)

        self.mode = self.get_parameter('localization_mode').value
        map_path = self.get_parameter('map_path').value
        self.x = self.get_parameter('initial_x').value
        self.y = self.get_parameter('initial_y').value
        self.yaw = self.get_parameter('initial_yaw').value
        self.odom_cov_scale = self.get_parameter('odom_cov_scale').value
        self.icp_max_dist = self.get_parameter('icp_max_distance').value

        self.get_logger().info(f'Localization Mode: {self.mode.upper()}')

        self.map_pcd = None
        if self.mode in ['fusion', 'scan']:
            if not map_path:
                self.get_logger().error('map_path parameter is required for scan/fusion mode!')
                raise RuntimeError('Missing map_path')
            self._load_map_as_pcd(map_path)

        self.scan_x, self.scan_y, self.scan_yaw = self.x, self.y, self.yaw
        self.odom_x, self.odom_y, self.odom_yaw = self.x, self.y, self.yaw
        self.odom_received = False
        self.last_odom_msg = None

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        qos = QoSProfile(depth=10)

        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'amcl_pose', qos
        )

        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_callback, qos
        )

        if self.mode in ['fusion', 'scan']:
            self.scan_sub = self.create_subscription(
                LaserScan, 'scan', self._scan_callback, qos_profile_sensor_data
            )

        self.get_logger().info('Scan-Odom Localization Node Initialized successfully.')

    def _resolve_yaml_path(self, map_path):
        """폴더 경로 또는 확장자 없는 파일 경로에서 .yaml 파일 자동 검색"""
        if os.path.isdir(map_path):
            yamls = glob.glob(os.path.join(map_path, '*.yaml'))
            if not yamls:
                raise FileNotFoundError(f"No .yaml file found inside directory: {map_path}")
            return yamls[0]
        
        yaml_file = map_path if map_path.endswith('.yaml') else map_path + '.yaml'
        if os.path.exists(yaml_file):
            return yaml_file
            
        # 폴더일 가능성 체크
        if os.path.exists(map_path):
            yamls = glob.glob(os.path.join(map_path, '*.yaml'))
            if yamls:
                return yamls[0]
                
        raise FileNotFoundError(f"Map yaml file not found for path: {map_path}")

    def _load_map_as_pcd(self, map_path):
        yaml_file = self._resolve_yaml_path(map_path)
        self.get_logger().info(f'Resolved Map YAML File: {yaml_file}')

        with open(yaml_file, 'r') as f:
            meta = yaml.safe_load(f)

        img_filename = meta['image']
        img_path = img_filename if os.path.isabs(img_filename) else os.path.join(os.path.dirname(yaml_file), img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Map image file not found: {img_path}")

        res = meta['resolution']
        origin = meta['origin']

        # PIL을 이용하여 PNG, PGM, JPG 모두 자동 지원
        img = Image.open(img_path).convert('L')
        img_arr = np.array(img)
        occ_r, occ_c = np.where(img_arr < 128)

        pts_x = origin[0] + occ_c * res
        pts_y = origin[1] + (img_arr.shape[0] - occ_r) * res
        pts_z = np.zeros_like(pts_x)

        map_pts = np.vstack((pts_x, pts_y, pts_z)).T
        self.map_pcd = o3d.geometry.PointCloud()
        self.map_pcd.points = o3d.utility.Vector3dVector(map_pts)
        self.get_logger().info(f'Loaded Map PointCloud ({img_filename}) with {len(map_pts)} points.')

    def _odom_callback(self, msg: Odometry):
        self.last_odom_msg = msg
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.odom_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.odom_received = True

        if self.mode == 'odom':
            self.x, self.y, self.yaw = self.odom_x, self.odom_y, self.odom_yaw
            self._publish_pose(msg.header.stamp)
            self._publish_tf(msg.header.stamp)

    def _scan_callback(self, msg: LaserScan):
        if not self.odom_received or self.map_pcd is None:
            return

        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        valid = (ranges >= msg.range_min) & (ranges <= msg.range_max)

        xs = ranges[valid] * np.cos(angles[valid])
        ys = ranges[valid] * np.sin(angles[valid])
        zs = np.zeros_like(xs)

        scan_pts = np.vstack((xs, ys, zs)).T
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(scan_pts)

        init_x = self.x
        init_y = self.y
        init_yaw = self.yaw

        init_trans = np.identity(4)
        init_trans[0, 3] = init_x
        init_trans[1, 3] = init_y
        init_trans[0, 0] = math.cos(init_yaw)
        init_trans[0, 1] = -math.sin(init_yaw)
        init_trans[1, 0] = math.sin(init_yaw)
        init_trans[1, 1] = math.cos(init_yaw)

        reg = o3d.pipelines.registration.registration_icp(
            scan_pcd, self.map_pcd, self.icp_max_dist, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        T = reg.transformation
        self.scan_x = T[0, 3]
        self.scan_y = T[1, 3]
        self.scan_yaw = math.atan2(T[1, 0], T[0, 0])

        if self.mode == 'scan':
            self.x, self.y, self.yaw = self.scan_x, self.scan_y, self.scan_yaw
        elif self.mode == 'fusion':
            scan_weight = 0.8
            odom_weight = 0.2
            self.x = scan_weight * self.scan_x + odom_weight * self.odom_x
            self.y = scan_weight * self.scan_y + odom_weight * self.odom_y
            self.yaw = math.atan2(
                scan_weight * math.sin(self.scan_yaw) + odom_weight * math.sin(self.odom_yaw),
                scan_weight * math.cos(self.scan_yaw) + odom_weight * math.cos(self.odom_yaw)
            )

        stamp = msg.header.stamp
        self._publish_pose(stamp)
        self._publish_tf(stamp)

    def _publish_pose(self, stamp):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.pose.position.x = float(self.x)
        pose_msg.pose.pose.position.y = float(self.y)

        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        pose_msg.pose.pose.orientation.x = qx
        pose_msg.pose.pose.orientation.y = qy
        pose_msg.pose.pose.orientation.z = qz
        pose_msg.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = 0.05 if self.mode != 'odom' else 0.5
        cov[7] = 0.05 if self.mode != 'odom' else 0.5
        cov[35] = 0.02 if self.mode != 'odom' else 0.2
        pose_msg.pose.covariance = cov

        self.pose_pub.publish(pose_msg)

    def _publish_tf(self, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        dx = self.x - self.odom_x
        dy = self.y - self.odom_y
        dyaw = self.yaw - self.odom_yaw

        t.transform.translation.x = float(dx)
        t.transform.translation.y = float(dy)

        qx, qy, qz, qw = yaw_to_quat(dyaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ScanOdomLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()