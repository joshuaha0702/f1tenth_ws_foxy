#!/usr/bin/env python3
import os
import sys
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

class MapSaver(Node):
    def __init__(self, output_prefix, topic_name='/map'):
        super().__init__('map_saver')
        self.output_prefix = output_prefix
        self.saved = False
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(OccupancyGrid, topic_name, self.callback, qos)
        self.get_logger().info(f"Subscribed to '{topic_name}', waiting for map data...")

    def callback(self, msg):
        self.get_logger().info(f"Map received ({msg.info.width}x{msg.info.height}, res={msg.info.resolution:.4f})! Saving...")
        img = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        pgm = np.full(img.shape, 205, dtype=np.uint8)
        pgm[img == 0] = 254
        pgm[img == 100] = 0
        pgm = np.flipud(pgm)

        base_dir = os.path.dirname(self.output_prefix)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
            
        file_basename = os.path.basename(self.output_prefix)

        # Save PGM image
        pgm_path = f"{self.output_prefix}.pgm"
        with open(pgm_path, 'wb') as f:
            f.write(f"P5\n{msg.info.width} {msg.info.height}\n255\n".encode())
            f.write(pgm.tobytes())

        # Save YAML metadata
        yaml_path = f"{self.output_prefix}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump({
                'image': f"{file_basename}.pgm",
                'resolution': float(msg.info.resolution),
                'origin': [float(msg.info.origin.position.x), float(msg.info.origin.position.y), 0.0],
                'occupied_thresh': 0.65,
                'free_thresh': 0.25,
                'negate': 0
            }, f, default_flow_style=False)

        self.get_logger().info(f"Saved {pgm_path} and {yaml_path}")
        self.saved = True
        raise SystemExit

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 save_map.py <output_prefix> [topic_name]")
        print("Example: python3 save_map.py /path/to/maps/track1/track1_map /map")
        sys.exit(1)

    output_prefix = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else '/map'

    rclpy.init()
    node = MapSaver(output_prefix, topic_name)
    try:
        # Wait up to 15 seconds for map
        timeout_sec = 15.0
        start_time = node.get_clock().now()
        while rclpy.ok() and not node.saved:
            rclpy.spin_once(node, timeout_sec=0.2)
            elapsed = (node.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout_sec:
                node.get_logger().error(f"Timeout ({timeout_sec}s) waiting for map on '{topic_name}'")
                sys.exit(1)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
