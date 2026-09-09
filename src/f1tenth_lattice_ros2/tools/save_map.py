import sys
import rclpy
import yaml
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy

class MapSaver(Node):
    def __init__(self, map_name):
        super().__init__('map_saver')
        self.map_name = map_name
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub = self.create_subscription(OccupancyGrid, '/map2d', self.callback, qos)
        self.get_logger().info('Waiting for /map2d ...')

    def callback(self, msg):
        self.get_logger().info(f'Map received! Saving to {self.map_name}...')
        img = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        pgm = np.full(img.shape, 205, dtype=np.uint8)
        pgm[img == 0] = 254
        pgm[img == 100] = 0
        pgm = np.flipud(pgm)

        base_path = f'/root/f1tenth_ws/src/f1tenth_lattice_ros2/maps/{self.map_name}/{self.map_name}'
        
        with open(f'{base_path}.pgm', 'wb') as f:
            f.write(f"P5\n{msg.info.width} {msg.info.height}\n255\n".encode())
            f.write(pgm.tobytes())

        with open(f'{base_path}.yaml', 'w') as f:
            yaml.dump({
                'image': f'{self.map_name}.pgm',
                'resolution': float(msg.info.resolution),
                'origin': [float(msg.info.origin.position.x), float(msg.info.origin.position.y), 0.0],
                'occupied_thresh': 0.65,
                'free_thresh': 0.25,
                'negate': 0
            }, f, default_flow_style=False)
            
        self.get_logger().info('Map saved successfully! Check the folder.')
        raise SystemExit

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 save_map.py <map_name>")
        sys.exit(1)
        
    rclpy.init()
    node = MapSaver(sys.argv[1])
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
