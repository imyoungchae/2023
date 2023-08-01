#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from numpy import linalg
from scipy.spatial.transform import Rotation as Rot
import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

global mat
mat = np.matrix

def ur_inv(T):
    d1, a2, a3 = 0.1625, -0.425, -0.39225
    d4, d5, d6 = 0.1333, 0.0997, 0.0996

    T06 = T
    T60 = np.linalg.inv(T06)

    A = d6 * T60[0, 2] - T60[0, 3]
    B = d6 * T60[1, 2] - T60[1, 3]
    C = T60[2, 3] - d1
    
    D1 = C * a3 + B * d4
    D2 = B * a3 - C * d4
    D3 = A * a3 - a2 * d4
    
    psi = np.arctan2(D2, D1)
    q1 = np.arctan2(C / np.cos(psi), A)

    q3 = np.arctan2(D1 / np.cos(q1), D3 / np.cos(q1))

    T01 = np.array([[np.cos(q1), -np.sin(q1), 0, 0],
                    [np.sin(q1), np.cos(q1), 0, 0],
                    [0, 0, 1, d1],
                    [0, 0, 0, 1]])
    
    T12 = np.array([[np.cos(-np.pi/2), 0, np.sin(-np.pi/2), 0],
                    [0, 1, 0, 0],
                    [-np.sin(-np.pi/2), 0, np.cos(-np.pi/2), 0],
                    [0, 0, 0, 1]])
    
    T23 = np.array([[np.cos(q3 + np.pi/2), np.sin(q3 + np.pi/2), 0, a2],
                    [-np.sin(q3 + np.pi/2), np.cos(q3 + np.pi/2), 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    
    T34 = np.array([[1, 0, 0, a3],
                    [0, -1, 0, -d4],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    
    T03 = T01 @ T12 @ T23 @ T34
    T36 = np.linalg.inv(T03) @ T06

    T_sym = np.array(
        [[0, -1, 0, d5],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    T63 = T_sym @ T36 @ T_sym
    T63R = T63[:3,:3]
    
    roll, pitch, yaw = Rot.from_matrix(T63R).as_euler('xyz')

    q4, q5, q6 = roll, pitch, yaw

    return q1, q3, q4, q5, q6


x = 0.387078
y = 4.149387
z = -2.302696
rx = -0.079900
ry = 0.064758
rz = -0.032299

r = Rot.from_euler('xyz', [rx, ry, rz], degrees=False)
rotation_matrix = r.as_matrix()
T = np.eye(4)
T[:3, :3] = rotation_matrix
T[:3, 3] = [x, y, z]

joint_angles = ur_inv(T)
print("Joint angles:", joint_angles)

class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'omni_data2',
            self.msg_callback,
            10)
        self.publisher = self.create_publisher(
            Float64MultiArray,
            'joint_angles',
            10)

    def msg_callback(self, msg):
        received_data = msg.data
        #self.get_logger().info('Received data: "%s"' % msg.data)

        x, y, z, rx, ry, rz = received_data

        # Check for NaN values before proceeding with the computation
        if any(math.isnan(val) for val in [x, y, z, rx, ry, rz]):
            self.get_logger().warning("Ignoring received message with NaN values.")
            return

        r = Rot.from_euler('xyz', [rx, ry, rz], degrees=False)
        rotation_matrix = r.as_matrix()

        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = [x, y, z]

        try:
            joint_angles =ur_inv(T)
            self.get_logger().info(f'Computed joint angles: {joint_angles}')
            self.publisher = self.create_publisher(Float64MultiArray, 'joint_angles', 10)
            self.get_logger().info('Publisher created and initialized for joint angles topic.')
            angles_msg = Float64MultiArray()
            # Ensure that the values are float before assigning to data
            angles = np.array(joint_angles).flatten().tolist()

            if all(isinstance(angle, float) for angle in angles):
                angles_msg.data = angles
            else:
                raise ValueError("Joint angles contain non-float values")

            self.publisher.publish(angles_msg)

        except ValueError as e:
            self.get_logger().warning("Error while computing joint angles: %s" % e)

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
