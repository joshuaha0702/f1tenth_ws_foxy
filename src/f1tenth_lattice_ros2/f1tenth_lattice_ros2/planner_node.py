import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

import numpy as np
import math
import time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from rcl_interfaces.msg import SetParametersResult
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
        self.declare_parameter('opponent_namespace', '')

        config_path = self.get_parameter('config_path').value
        raceline_path = self.get_parameter('raceline_path').value
        map_path = self.get_parameter('map_path').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steer = self.get_parameter('max_steering_angle').value
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
        # 직전 plan() 호출 시각(ROS Time). 선두 차량 속도 추정용 dt를 측정하는 데 사용한다.
        # (planning이 scan 트리거라 주기가 일정하지 않으므로 실측 간격을 넘겨줌)
        # use_sim_time=True 환경에서는 wall-clock이 아니라 sim 시간(/clock)을 따라야 하므로
        # time.perf_counter()가 아니라 self.get_clock()을 사용한다.
        self._last_plan_time = None

        # 에피소드 모드: 매니저에게 STOP/START 명령을 받아 publish gate를 토글함
        # (텔레포트 직후 stale state가 다음 에피소드로 새는 것을 차단)
        self._publishing_enabled = True

        # QoS
        qos = QoSProfile(depth=10)

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_callback, qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self._scan_callback, qos_profile_sensor_data
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

        # 에피소드 매니저로부터 STOP / START 신호 수신 (네임스페이스 무관 글로벌 토픽)
        self.create_subscription(
            String, '/episode/control', self._episode_control_callback, 10
        )

        # traj_v_scale을 런타임 변경 가능한 파라미터로 노출 (외부에서 `ros2 param set` 또는
        # 매니저의 SetParameters 호출로 갱신 가능). 기본값은 lattice_config.yaml의 값.
        self.declare_parameter('traj_v_scale', float(self.planner.traj_v_scale))
        # 런치 시점에 CLI로 override 됐을 수도 있으니 다시 읽어 동기화
        self.planner.traj_v_scale = float(self.get_parameter('traj_v_scale').value)
        self.add_on_set_parameters_callback(self._on_param_set)

        # Publish raceline visualization once after init
        self.create_timer(2.0, self._publish_raceline_once)
        self._raceline_published = False

        self.get_logger().info(
            f'Lattice planner node ready (scan-triggered planning, '
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

        if getattr(self, 'best_traj', None) is not None:
            # Pure pursuit on the selected local trajectory at high frequency
            steering, speed = self.planner.tracker.plan(
                self.pose_x, self.pose_y, self.pose_theta,
                self.velocity, self.best_traj
            )

            # Clamp outputs
            steering = float(np.clip(steering, -self.max_steer, self.max_steer))
            speed = float(np.clip(speed, 0.0, self.max_speed))

            drive_msg = AckermannDriveStamped()
            # 타임스탬프 = 이 drive 메시지를 실제로 발행하는 시점
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = steering
            self.drive_pub.publish(drive_msg)

    def _opp_odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.opp_pose = np.array([[x, y, theta]])

    def _on_param_set(self, params):
        """외부에서 SetParameters 서비스로 들어오는 파라미터 변경을 처리.

        현재는 traj_v_scale 만 동적 변경 허용. 그 외 파라미터는 변경을 거부하지 않고
        그냥 통과시킴(필요해지면 추가).
        """
        for p in params:
            if p.name == 'traj_v_scale':
                try:
                    new_val = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'traj_v_scale must be a number, got {p.value!r}',
                    )
                old_val = self.planner.traj_v_scale
                self.planner.traj_v_scale = new_val
                self.get_logger().info(
                    f'[param] traj_v_scale: {old_val} -> {new_val:.3f}'
                )
        return SetParametersResult(successful=True)

    def _episode_control_callback(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd == 'STOP':
            # 0속도 명령 한 번 박아두어 Gazebo plugin의 마지막 cmd_vel을 해제함
            stop_msg = AckermannDriveStamped()
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link'
            stop_msg.drive.speed = 0.0
            stop_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(stop_msg)
            # 다음 에피소드에 stale state가 새지 않도록 클리어
            self.odom_received = False
            self.opp_pose = np.empty((0, 3))
            self.best_traj = None
            # 다음 에피소드 첫 plan()에서 에피소드 사이의 긴 공백이 dt로 새지 않도록 리셋
            self._last_plan_time = None
            self._publishing_enabled = False
            self.get_logger().info('[episode] STOP — paused publishing & cleared state')
        elif cmd == 'START':
            self._publishing_enabled = True
            self.get_logger().info('[episode] START — resumed publishing')
        else:
            self.get_logger().warn(f'[episode] unknown control cmd: {msg.data!r}')

    def _scan_callback(self, msg: LaserScan):
        if not self._publishing_enabled:
            return
        if not self.odom_received:
            return

        opp_poses = self.opp_pose

        try:
            t0 = time.perf_counter()
            # 직전 plan() 호출과의 경과 시간 = 선두 차량 속도 추정용 dt.
            # sim 시간(use_sim_time)을 따르도록 ROS 클럭으로 측정한다.
            # 첫 호출이면 None을 넘겨 planner가 설정값 기반 기본값으로 폴백하게 한다.
            now = self.get_clock().now()
            if self._last_plan_time is not None:
                dt = (now - self._last_plan_time).nanoseconds * 1e-9
                if dt <= 0.0:
                    dt = None
            else:
                dt = None
            self._last_plan_time = now
            best_traj, best_cost, traj_cost, abs_v_cost, collision_cost = self.planner.plan(
                self.pose_x, self.pose_y, self.pose_theta,
                opp_poses, self.velocity, dt=dt
            )
            elapsed = time.perf_counter() - t0
            if elapsed > 0.03:
                timing = self.planner.last_timing
                if timing is not None:
                    clothoid_ms, eval_ms = timing
                    self.get_logger().warn(
                        f'[plan timing] total={elapsed*1000:.1f}ms '
                        f'(clothoid={clothoid_ms:.1f}ms, eval={eval_ms:.1f}ms)',
                        throttle_duration_sec=1.0
                    )
                else:
                    self.get_logger().warn(f'[plan timing] {elapsed*1000:.1f}ms', throttle_duration_sec=1.0)
            self.best_traj = best_traj
        except Exception as e:
            self.get_logger().warn(f'Lattice plan failed: {e}', throttle_duration_sec=2.0)
            return

        self._publish_best_traj(self.best_traj, msg.header.stamp)

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
