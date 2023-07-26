#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'omni_data',
            self.msg_callback,
            10)

    def msg_callback(self, msg):
        received_data = msg.data
        self.get_logger().info('Received data: "%s"' % msg.data)
        # Other processing if needed

def main(args=None):
    rclpy.init(args=args)
    pose_subscriber = PoseSubscriber()
    try:
        rclpy.spin(pose_subscriber)
    except KeyboardInterrupt:
        pass

    pose_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
