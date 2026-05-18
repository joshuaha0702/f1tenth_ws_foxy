#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist

class AckermannToTwist(Node):
    def __init__(self):
        super().__init__('ackermann_to_twist')
        self.sub = self.create_subscription(
            AckermannDriveStamped,
            'drive',
            self.callback,
            10)
        self.pub = self.create_publisher(Twist, 'drive_twist', 10)

    def callback(self, msg):
        twist = Twist()
        twist.linear.x = msg.drive.speed
        twist.angular.z = msg.drive.steering_angle
        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AckermannToTwist()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
