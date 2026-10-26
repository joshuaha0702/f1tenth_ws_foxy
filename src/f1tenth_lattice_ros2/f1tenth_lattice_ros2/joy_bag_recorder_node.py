#!/usr/bin/env python3

import os
import signal
import subprocess
import yaml
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


class JoyBagRecorderNode(Node):
    """
    조이스틱 버튼 입력에 따라 ROS2 Bag 녹화를 시작하고 중지하는 노드.
    
    기본 설정:
      - 7번 버튼 (start_button): 녹화 시작
      - 6번 버튼 (stop_button) : 녹화 중지
    
    녹화 대상 토픽:
      - /scan (LiDAR)
      - /odom (Odometry)
      - /drive (AI 모델 / 플래너 출력 제어)
      - /teleop (조이스틱 수동 입력 제어)
      - /ackermann_drive (Mux 통과 후 실제 차량에 전달된 최종 출력)
      (주의: /tf 는 고주파 대역폭을 과도하게 차지하여 레코더 큐 드롭을 유발하므로 기본 제외)
    """

    def __init__(self):
        super().__init__('joy_bag_recorder')

        # Declare parameters
        self.declare_parameter('start_button', 7)  # 기본값 7번 버튼 (녹화 시작)
        self.declare_parameter('stop_button', 6)   # 기본값 6번 버튼 (녹화 종료)
        self.declare_parameter('bag_prefix', 'f1tenth_run')
        self.declare_parameter('output_dir', 'bags')
        self.declare_parameter('record_queue_depth', 1000)
        self.declare_parameter('record_tf', False)

        self.start_button = self.get_parameter('start_button').value
        self.stop_button = self.get_parameter('stop_button').value
        self.bag_prefix = self.get_parameter('bag_prefix').value
        self.output_dir = self.get_parameter('output_dir').value
        self.record_queue_depth = int(self.get_parameter('record_queue_depth').value)
        self.record_tf = bool(self.get_parameter('record_tf').value)

        # 핵심 녹화 토픽 설정
        self.record_topics = [
            '/scan',
            '/odom',
            '/drive',
            '/teleop',
            '/ackermann_drive'
        ]
        if self.record_tf:
            self.record_topics.extend(['/tf', '/tf_static'])

        # Foxy rosbag2 큐 부족으로 인한 패킷 드롭(주기 2배화 버그) 방지용 QoS 프로파일 생성
        self.qos_profile_path = os.path.join(self.output_dir, 'joy_record_qos.yaml')
        self._ensure_qos_profile()

        self.is_recording = False
        self.recording_proc = None
        self.prev_start_state = 0
        self.prev_stop_state = 0

        # Subscribe to /joy
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        self.get_logger().info("==========================================================")
        self.get_logger().info(f"🎮 [JoyBagRecorder] Ready.")
        self.get_logger().info(f"   - [7번 버튼] 누르면 -> 녹화 시작 (START)")
        self.get_logger().info(f"   - [6번 버튼] 누르면 -> 녹화 중지 (STOP)")
        self.get_logger().info(f"   - 녹화 토픽: {', '.join(self.record_topics)}")
        self.get_logger().info(f"   - QoS 수신 큐 깊이: {self.record_queue_depth} (드롭 방지 적용)")
        self.get_logger().info("==========================================================")

    def _ensure_qos_profile(self):
        os.makedirs(self.output_dir, exist_ok=True)
        qos_dict = {
            topic: {
                'history': 'keep_last',
                'depth': self.record_queue_depth,
                'reliability': 'reliable',
                'durability': 'volatile'
            }
            for topic in self.record_topics
        }
        with open(self.qos_profile_path, 'w') as f:
            yaml.safe_dump(qos_dict, f)

    def joy_callback(self, msg: Joy):
        max_btn_idx = max(self.start_button, self.stop_button)
        if len(msg.buttons) <= max_btn_idx:
            return

        curr_start = msg.buttons[self.start_button]
        curr_stop = msg.buttons[self.stop_button]

        # 7번 버튼 감지 -> 녹화 시작
        if curr_start == 1 and self.prev_start_state == 0:
            if not self.is_recording:
                self.start_recording()
            else:
                self.get_logger().warn("⚠️ [JoyBagRecorder] 이미 녹화가 진행 중입니다!")

        # 6번 버튼 감지 -> 녹화 중지
        if curr_stop == 1 and self.prev_stop_state == 0:
            if self.is_recording:
                self.stop_recording()
            else:
                self.get_logger().info("ℹ️ [JoyBagRecorder] 진행 중인 녹화가 없습니다.")

        self.prev_start_state = curr_start
        self.prev_stop_state = curr_stop

    def start_recording(self):
        # 타임스탬프 기반 저장 경로 설정
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_name = f"{self.bag_prefix}_{timestamp}"

        if not os.path.isabs(self.output_dir):
            base_dir = os.getcwd()
            bag_path = os.path.join(base_dir, self.output_dir, bag_name)
        else:
            bag_path = os.path.join(self.output_dir, bag_name)

        os.makedirs(os.path.dirname(bag_path), exist_ok=True)
        self._ensure_qos_profile()

        cmd = [
            'ros2', 'bag', 'record',
            '-o', bag_path,
            '--qos-profile-overrides-path', os.path.abspath(self.qos_profile_path)
        ] + self.record_topics

        self.get_logger().info("==========================================================")
        self.get_logger().warn(f"🔴 [REC START] ROS2 Bag 녹화가 시작되었습니다!")
        self.get_logger().warn(f"📂 저장 경로: {bag_path}")
        self.get_logger().info("==========================================================")

        try:
            # preexec_fn=os.setsid를 통해 별도 프로세스 그룹으로 띄워야
            # SIGINT 전달 시 하위 실제 레코더 C++ 바이너리까지 깔끔하게 종료됨
            self.recording_proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.is_recording = True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to start ros2 bag record: {e}")

    def stop_recording(self):
        if self.recording_proc and self.is_recording:
            self.get_logger().info("⬛ [REC STOP] ROS2 Bag 녹화 종료 처리 중...")
            try:
                # 프로세스 그룹 전체에 SIGINT 전달 (자식 레코더 정상 종료 및 DB 플러시)
                os.killpg(os.getpgid(self.recording_proc.pid), signal.SIGINT)
                self.recording_proc.wait(timeout=7.0)
            except subprocess.TimeoutExpired:
                self.get_logger().warn("⚠️ SIGINT timeout expired, sending SIGTERM to process group...")
                try:
                    os.killpg(os.getpgid(self.recording_proc.pid), signal.SIGTERM)
                    self.recording_proc.wait(timeout=3.0)
                except Exception:
                    pass
            except ProcessLookupError:
                pass
            except Exception as e:
                self.get_logger().error(f"❌ Error stopping recorder process: {e}")
            finally:
                self.recording_proc = None
                self.is_recording = False
                self.get_logger().info("==========================================================")
                self.get_logger().info("✅ [REC SAVED] ROS2 Bag 데이터가 안전하게 저장되었습니다.")
                self.get_logger().info("==========================================================")

    def destroy_node(self):
        if self.is_recording:
            self.stop_recording()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JoyBagRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
