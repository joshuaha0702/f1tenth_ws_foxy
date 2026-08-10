import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import math
import time
import threading

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64MultiArray, MultiArrayDimension, String
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Point, PoseStamped

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
        self.best_traj = None

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
        self._plan_generation = 0

        # odom/Pure Pursuit와 scan/lattice planning을 별도 스레드에서 실행한다.
        self._control_group = MutuallyExclusiveCallbackGroup()
        self._planning_group = MutuallyExclusiveCallbackGroup()

        # 두 callback group이 공유하는 차량 상태와 trajectory를 보호한다.
        self._state_lock = threading.Lock()
        self._trajectory_lock = threading.Lock()

        # QoS
        qos = QoSProfile(depth=10)
        # 제어 콜백이 밀릴 경우 과거 odom을 처리하지 않고 최신 상태를 사용한다.
        odom_qos = QoSProfile(depth=1)

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_callback, odom_qos,
            callback_group=self._control_group
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self._scan_callback, qos_profile_sensor_data,
            callback_group=self._planning_group
        )
        if opponent_ns:
            self.create_subscription(
                Odometry, f'/{opponent_ns}/odom', self._opp_odom_callback, qos,
                callback_group=self._control_group
            )
            self.get_logger().info(f'Head-to-head mode: tracking opponent /{opponent_ns}/odom')

        self.trajectory_pub = self.create_publisher(
            Float64MultiArray, 'planned_trajectory', QoSProfile(depth=1)
        )
        self.raceline_pub = self.create_publisher(MarkerArray, 'raceline_marker', qos)
        self.best_traj_pub = self.create_publisher(Marker, 'best_traj_marker', qos)

        # 에피소드 매니저로부터 STOP / START 신호 수신 (네임스페이스 무관 글로벌 토픽)
        self.create_subscription(
            String, '/episode/control', self._episode_control_callback, 10,
            callback_group=self._control_group
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
        q = msg.pose.pose.orientation
        pose_x = msg.pose.pose.position.x
        pose_y = msg.pose.pose.position.y
        pose_theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        velocity = msg.twist.twist.linear.x

        with self._state_lock:
            self.pose_x = pose_x
            self.pose_y = pose_y
            self.pose_theta = pose_theta
            self.velocity = velocity
            self.latest_odom_stamp = msg.header.stamp
            self.odom_received = True

    def _opp_odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = quat_to_yaw(q.x, q.y, q.z, q.w)
        with self._state_lock:
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
            # 다음 에피소드에 stale state가 새지 않도록 클리어
            with self._state_lock:
                self.odom_received = False
                self.opp_pose = np.empty((0, 3))
                self._last_plan_time = None
                self._publishing_enabled = False
                # 진행 중이던 scan planning 결과도 무효화한다.
                self._plan_generation += 1
            with self._trajectory_lock:
                self.best_traj = None
            self.trajectory_pub.publish(Float64MultiArray())
            self.get_logger().info('[episode] STOP — paused publishing & cleared state')
        elif cmd == 'START':
            with self._state_lock:
                self._publishing_enabled = True
            self.get_logger().info('[episode] START — resumed publishing')
        else:
            self.get_logger().warn(f'[episode] unknown control cmd: {msg.data!r}')

    def _scan_callback(self, msg: LaserScan):
        # Planning 시작 시점의 상태를 snapshot하여 odom 스레드와 독립적으로 계산한다.
        with self._state_lock:
            if not self._publishing_enabled or not self.odom_received:
                return
            pose_x = self.pose_x
            pose_y = self.pose_y
            pose_theta = self.pose_theta
            velocity = self.velocity
            opp_poses = self.opp_pose.copy()
            plan_generation = self._plan_generation

            # 직전 plan() 호출과의 경과 시간 = 선두 차량 속도 추정용 dt.
            now = self.get_clock().now()
            if self._last_plan_time is not None:
                dt = (now - self._last_plan_time).nanoseconds * 1e-9
                if dt <= 0.0:
                    dt = None
            else:
                dt = None
            self._last_plan_time = now

        try:
            t0 = time.perf_counter()
            best_traj, best_cost, traj_cost, abs_v_cost, collision_cost = self.planner.plan(
                pose_x, pose_y, pose_theta,
                opp_poses, velocity, dt=dt
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
            # STOP 이후 완료된 오래된 planning 결과는 다음 에피소드로 넘기지 않는다.
            with self._state_lock:
                plan_is_valid = (
                    self._publishing_enabled
                    and plan_generation == self._plan_generation
                )
            if not plan_is_valid:
                return
            with self._trajectory_lock:
                self.best_traj = best_traj
        except Exception as e:
            self.get_logger().warn(f'Lattice plan failed: {e}', throttle_duration_sec=2.0)
            return

        self._publish_trajectory(best_traj)
        self._publish_best_traj(best_traj, msg.header.stamp)

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
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
