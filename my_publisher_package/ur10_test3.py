#!/usr/bin/env python3
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import copy
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from numpy import linalg
from scipy.spatial.transform import Rotation as Rot
import cmath
from math import acos, atan2, cos, pi, sin, sqrt,asin,pi
from math import acos as acos

gym = gymapi.acquire_gym()
axes_geom = gymutil.AxesGeometry(0.1)
joint_angles = np.zeros(6, dtype=np.float32)
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))
MAT = np.matrix
args = gymutil.parse_arguments(
    description="UR10 Robot Simulation")
args.physics_engine = gymapi.SIM_PHYSX

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 3
sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.thickness = 1
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

table_dims=gymapi.Vec3(0.5,1.8,0.5)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(1.0, 1.5, 0)
object_pose = gymapi.Transform()
table_x = table_pose.p.x
table_y = table_pose.p.y
table_z = table_pose.p.z
table_height = table_dims.y

table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer,None,gymapi.Vec3(3,3.8,3),gymapi.Vec3(1.1,1.8,-0.15))

if viewer is None:
    print("*** Failed to create viewer")
    quit()

asset_root = "/home/son/my_ws/src/my_publisher_package/src/isaacgym/assets"
urdf_file = "urdf/ur10/ur10_test.urdf"

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False

num_envs = 1
env_lower = gymapi.Vec3(0, 0, 0)
env_upper = gymapi.Vec3(3, 3, 3)
env = gym.create_env(sim,env_lower, env_upper, num_envs)

ur_pose = gymapi.Transform()
ur_base_height = 1.2 
ur_pose.p = gymapi.Vec3(1.1,1.8,-0.15)

ur_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
ur_actor = gym.create_actor(env, asset, ur_pose, "ur10", 0, 1)
table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

TRANSL_PARAM_Z = np.array([0.1273, 0, 0, 0, 0.163941, 0.1157, 0.0922])
TRANSL_PARAM_X = np.array([0, 0, -0.612, -0.5723, 0, 0, 0])
ROT_PARAM_X = np.array([0, np.pi/2, 0, 0, 0, np.pi/2, -np.pi/2])

DH_matrix_UR10 = np.matrix([[0, pi / 2.0, 0.1273],
                            [-0.612, 0, 0],
                            [-0.5723, 0, 0],
                            [0, pi / 2.0, 0.163941],
                            [0, -pi / 2.0, 0.1157],
                            [0, 0, 0.0922]])

def mat_transtorm_DH(DH_matrix, n, edges=np.matrix([[0], [0], [0], [0], [0], [0]])):
    n = n - 1
    t_z_theta = np.matrix([[cos(edges[n]), -sin(edges[n]), 0, 0],
                           [sin(edges[n]), cos(edges[n]), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], copy=False)
    t_zd = np.matrix(np.identity(4), copy=False)
    t_zd[2, 3] = DH_matrix[n, 2]
    t_xa = np.matrix(np.identity(4), copy=False)
    t_xa[0, 3] = DH_matrix[n, 0]
    t_x_alpha = np.matrix([[1, 0, 0, 0],
                           [0, cos(DH_matrix[n, 1]), -sin(DH_matrix[n, 1]), 0],
                           [0, sin(DH_matrix[n, 1]), cos(DH_matrix[n, 1]), 0],
                           [0, 0, 0, 1]], copy=False)
    transform = t_z_theta * t_zd * t_xa * t_x_alpha
    return transform


def htm_rotation_around_x(angle: float) -> np.matrix:
    return np.matrix([[1, 0, 0, 0],
                     [0, cos(angle), -sin(angle), 0],
                     [0, sin(angle), cos(angle), 0],
                     [0, 0, 0, 1]])

def htm_rotation_around_y(angle: float) -> np.matrix:
    return np.matrix([[cos(angle), 0, sin(angle), 0],
                     [0, 1, 0, 0],
                     [-sin(angle), 0, cos(angle), 0],
                     [0, 0, 0, 1]])

def htm_rotation_around_z(angle: float) -> np.matrix:
    return np.matrix([[cos(angle), -sin(angle), 0, 0],
                     [sin(angle), cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def htm_translation(translation_vector: np.array) -> np.matrix:
    return np.matrix([[1, 0, 0, translation_vector[0]],
                     [0, 1, 0, translation_vector[1]],
                     [0, 0, 1, translation_vector[2]],
                     [0, 0, 0, 1]])

def get_desired_pose_htm(
    position: np.array,
    roll: float,
    pitch: float,
    yaw: float
):
    rot_x = htm_rotation_around_x(roll)
    rot_y = htm_rotation_around_y(pitch)
    rot_z = htm_rotation_around_z(yaw)
    desired_pose = rot_z * rot_y * rot_x
    position_vector = position.reshape(3, 1)
    desired_pose[:3, -1] = position_vector
    return desired_pose

def htm_base_link_inertia_to_shoulder_link(theta1: float) -> np.matrix:
    translation_z = htm_translation([0, 0, TRANSL_PARAM_Z[0]])
    rotation_z = htm_rotation_around_z(theta1)
    htm_01 = translation_z * rotation_z
    return htm_01


def htm_shoulder_link_to_upper_arm_link(theta2: float) -> np.matrix:
    rotation_x = htm_rotation_around_x(ROT_PARAM_X[1])
    rotation_z = htm_rotation_around_z(theta2)
    htm_12 = rotation_x * rotation_z
    return htm_12


def htm_upper_arm_link_to_forearm_link(theta3: float) -> np.matrix:
    translation_x = htm_translation([TRANSL_PARAM_X[2], 0, 0])
    rotation_z = htm_rotation_around_z(theta3)
    htm_23 = translation_x * rotation_z
    return htm_23


def htm_forearm_link_to_forearm_link_x() -> np.matrix:
    htm_34 = htm_translation([TRANSL_PARAM_X[3], 0, 0])
    return htm_34


def htm_forearm_link_x_to_wrist_1_link(theta4: float) -> np.matrix:
    translation_z = htm_translation([0, 0, TRANSL_PARAM_Z[4]])
    rotation_z = htm_rotation_around_z(theta4)
    htm_45 = translation_z * rotation_z
    return htm_45


def htm_wrist_1_link_to_wrist_2_link(theta5: float) -> np.matrix:
    rotation_x = htm_rotation_around_x(ROT_PARAM_X[5])
    translation_z = htm_translation([0, 0, TRANSL_PARAM_X[5]])
    rotation_z = htm_rotation_around_z(theta5)
    htm_56 = rotation_x * translation_z * rotation_z
    return htm_56


def htm_wrist_2_link_to_wrist_3_link(theta6: float) -> np.matrix:
    rotation_x = htm_rotation_around_x(ROT_PARAM_X[-1])
    translation_z = htm_translation([0, 0, TRANSL_PARAM_Z[-1]])
    rotation_z = htm_rotation_around_z(theta6)
    htm_67 = rotation_x * translation_z * rotation_z
    return htm_67


def get_theta1(
    solutions: np.array,
    desired_pose_07: np.array
) -> np.matrix:
    position_link_89 =\
        desired_pose_07 * MAT([0, 0, -TRANSL_PARAM_Z[6], 1]).T \
        - MAT([0, 0, 0, 1]).T
    beta = atan2(position_link_89[1, 0], position_link_89[0, 0])
    position_wrist_2_xy = sqrt(position_link_89[1, 0]**2 +
                               position_link_89[0, 0]**2)

    if TRANSL_PARAM_Z[4] > position_wrist_2_xy:
        raise ValueError("d4 cannot be higher than position_wrist_2_xy."
                         "No solution for theta1")

    gamma = acos(TRANSL_PARAM_Z[4] / position_wrist_2_xy)
    solutions[0, 0:4] = pi/2 + beta + gamma
    solutions[0, 4:8] = pi/2 + beta - gamma

    return solutions


def get_theta5(
    solutions: np.array,
    desired_pose_07: np.array
) -> np.matrix:
    wrist_up_or_down_configs = [0, 4]  
    for config in wrist_up_or_down_configs:
        htm_link_01 = htm_base_link_inertia_to_shoulder_link(
            solutions[0, config])

        htm_link_12 = htm_shoulder_link_to_upper_arm_link(0)
        htm_link_02 = htm_link_01 * htm_link_12
        htm_link_20 = linalg.inv(htm_link_02)
        htm_link_27 = htm_link_20 * desired_pose_07

        acos_num = htm_link_27[2, 3]-TRANSL_PARAM_Z[4]
        acos_den = TRANSL_PARAM_Z[6]

        if acos_num > acos_den:
            raise ValueError("P16z - d4 cannot be higher than d6."
                             "\nIn other words, the z axis of wrist_3_link"
                             "cannot be parallel to the wrist_2_link,"
                             " wrist_1_link, forearm_link and upper_arm_link"
                             "z axis.")

        theta_5 = acos(acos_num/acos_den)
        solutions[4, config:config+2] = theta_5
        solutions[4, config+2:config+4] = -theta_5

    return solutions


def get_theta6(
    solutions: np.array,
    desired_pose_07: np.array
) -> np.matrix:
    configs = [0, 2, 4, 6]
    for config in configs:
        htm_link_01 = htm_base_link_inertia_to_shoulder_link(
            solutions[0, config])
        htm_link_12 = htm_shoulder_link_to_upper_arm_link(0)
        htm_link_02 = htm_link_01 * htm_link_12
        htm_link_20 = linalg.inv(htm_link_02)
        htm_link_27 = htm_link_20 * desired_pose_07
        htm_link_72 = linalg.inv(htm_link_27)
        theta5 = solutions[4, config]
        sin_theta5 = sin(theta5)
        theta6 = atan2(-htm_link_72[1, 2] / sin_theta5,
                       htm_link_72[0, 2] / sin_theta5)
        solutions[5, config:config+2] = theta6

    return solutions


def get_theta3(
    solutions: np.array,
    desired_pose_07: np.array
) -> np.matrix:
    configs = [0, 2, 4, 6]
    for config in configs:
        theta1 = solutions[0, config]
        htm_link_01 = htm_base_link_inertia_to_shoulder_link(theta1)
        htm_link_12 = htm_shoulder_link_to_upper_arm_link(0)
        htm_link_02 = htm_link_01 * htm_link_12
        htm_link_20 = linalg.inv(htm_link_02)
        htm_link_27 = htm_link_20 * desired_pose_07
        theta6 = solutions[5, config]
        htm_link_67 = htm_wrist_2_link_to_wrist_3_link(theta6)
        theta5 = solutions[4, config]
        htm_link_56 = htm_wrist_1_link_to_wrist_2_link(theta5)
        htm_link_25 = htm_link_27 * linalg.inv(htm_link_56 * htm_link_67)
        position_24 = htm_link_25 * \
            MAT([0, 0, -TRANSL_PARAM_Z[4], 1]).T - MAT([0, 0, 0, 1]).T

        theta3 = cmath.acos((
            linalg.norm(position_24)**2
            - TRANSL_PARAM_X[2]**2 - TRANSL_PARAM_X[3]**2) /
            (2 * TRANSL_PARAM_X[2] * TRANSL_PARAM_X[3]))

        solutions[2, config] = theta3.real
        solutions[2, config+1] = -theta3.real

    return solutions


def get_theta2(
    solutions: np.array,
    desired_pose_07: np.array
) -> np.matrix:
    configs = [0, 1, 2, 3, 4, 5, 6, 7]
    for config in configs:
        theta1 = solutions[0, config]
        htm_link_01 = htm_base_link_inertia_to_shoulder_link(theta1)
        htm_link_12 = htm_shoulder_link_to_upper_arm_link(0)
        htm_link_02 = htm_link_01 * htm_link_12

        htm_link_20 = linalg.inv(htm_link_02)
        htm_link_27 = htm_link_20 * desired_pose_07

        theta6 = solutions[5, config]
        htm_link_67 = htm_wrist_2_link_to_wrist_3_link(theta6)
        theta5 = solutions[4, config]
        htm_link_56 = htm_wrist_1_link_to_wrist_2_link(theta5)

        htm_link_25 = htm_link_27 * linalg.inv(htm_link_56 * htm_link_67)

        position_24 = htm_link_25 * \
            MAT([0, 0, -TRANSL_PARAM_Z[4], 1]).T - MAT([0, 0, 0, 1]).T

        gamma = atan2(position_24[1, 0], position_24[0, 0])
        theta3 = solutions[2, config]
        beta = atan2(TRANSL_PARAM_X[3]*sin(theta3),
                     TRANSL_PARAM_X[2] + TRANSL_PARAM_X[3]*cos(theta3))
        theta2 = gamma - beta
        solutions[1, config] = theta2

    return solutions


def get_theta4(
    solutions: np.array,
    desired_pose_07: np.array
) -> np.matrix:
    configs = [0, 1, 2, 3, 4, 5, 6, 7]
    for config in configs:
        theta1 = solutions[0, config]
        htm_link_01 = htm_base_link_inertia_to_shoulder_link(theta1)
        theta2 = solutions[1, config]
        htm_link_12 = htm_shoulder_link_to_upper_arm_link(theta2)
        htm_link_02 = htm_link_01 * htm_link_12
        htm_link_20 = linalg.inv(htm_link_02)
        htm_link_27 = htm_link_20 * desired_pose_07

        theta6 = solutions[5, config]
        htm_link_67 = htm_wrist_2_link_to_wrist_3_link(theta6)
        theta5 = solutions[4, config]
        htm_link_56 = htm_wrist_1_link_to_wrist_2_link(theta5)

        htm_link_25 = htm_link_27 * linalg.inv(htm_link_56 * htm_link_67)

        htm_link_34 = htm_forearm_link_to_forearm_link_x()
        theta3 = solutions[2, config]
        htm_link_23 = htm_upper_arm_link_to_forearm_link(theta3)
        htm_link_45 = linalg.inv(htm_link_34) * \
            linalg.inv(htm_link_23) * htm_link_25
        theta4 = atan2(htm_link_45[1, 0], htm_link_45[0, 0])

        solutions[3, config] = theta4

    return solutions


def inverse_kinematics(
    desired_pose_07: np.array,
    print_debug: bool = True,
    solution_index: int = 5
) -> np.matrix:

    solutions = MAT(np.zeros((6, 8)))
    solutions = get_theta1(solutions, desired_pose_07)
    solutions = get_theta5(solutions, desired_pose_07)
    solutions = get_theta6(solutions, desired_pose_07)
    solutions = get_theta3(solutions, desired_pose_07)
    solutions = get_theta2(solutions, desired_pose_07)
    solutions = get_theta4(solutions, desired_pose_07)

    if print_debug:
        solution_deg = np.degrees(solutions).astype(int)
        print(f"Inverse Kinematics Solutions [degrees|int]:\n{solution_deg}")
    solutions = np.array(solutions.real[:, solution_index]).reshape(6)
    return solutions


class UR_RPY_Position:
    def __init__(self, X, Y, Z, RX,RY,RZ):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.RX= RX
        self.RY = RY
        self.RZ = RZ

class UR_TCP_Position:
    def __init__(self, X, Y, Z, RX,RY,RZ):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.RX= RX
        self.RY = RY
        self.RZ = RZ

class Transform_M:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.r11 = 0
        self.r12 = 0
        self.r13 = 0
        self.r21 = 0
        self.r22 = 0
        self.r23 = 0
        self.r31 = 0
        self.r32 = 0
        self.r33 = 0
        self.wn = 0
        self.wo = 0
        self.wa = 0
        self.w = 1

def RPY_position2transform(input_position: UR_RPY_Position, output_transform: Transform_M):
    print("RPY_position2transform()")

    output_transform.x = input_position.X
    output_transform.y = input_position.Y
    output_transform.z = input_position.Z

    pi_a = input_position.RX
    pi_o = input_position.RY
    pi_n = input_position.RZ

    output_transform.r11 = math.cos(pi_a) * math.cos(pi_o)
    output_transform.r12 = (math.cos(pi_a) * math.sin(pi_o) * math.sin(pi_n)) - (math.sin(pi_a) * math.cos(pi_n))
    output_transform.r13 = (math.cos(pi_a) * math.sin(pi_o) * math.cos(pi_n)) + (math.sin(pi_a) * math.sin(pi_n))

    output_transform.r21 = math.sin(pi_a) * math.cos(pi_o)
    output_transform.r22 = (math.sin(pi_a) * math.sin(pi_o) * math.sin(pi_n)) + (math.cos(pi_a) * math.cos(pi_n))
    output_transform.r23 = (math.sin(pi_a) * math.sin(pi_o) * math.cos(pi_n)) - (math.cos(pi_a) * math.sin(pi_n))

    output_transform.r31 = -math.sin(pi_o)
    output_transform.r32 = math.cos(pi_o) * math.sin(pi_n)
    output_transform.r33 = math.cos(pi_o) * math.cos(pi_n)

    output_transform.wn = 0
    output_transform.wo = 0
    output_transform.wa = 0
    output_transform.w = 1

    print(" ")
    print("*********************output_transform***********************")
    print(f"x: {output_transform.x}")
    print(f"y: {output_transform.y}")
    print(f"z: {output_transform.z}")
    print(f"r11: {output_transform.r11}")
    print(f"r12: {output_transform.r12}")
    print(f"r13: {output_transform.r13}")
    print(f"r21: {output_transform.r21}")
    print(f"r22: {output_transform.r22}")
    print(f"r23: {output_transform.r23}")
    print(f"r31: {output_transform.r31}")
    print(f"r32: {output_transform.r32}")
    print(f"r33: {output_transform.r33}")
    print(f"wn: {output_transform.wn}")
    print(f"wo: {output_transform.wo}")
    print(f"wa: {output_transform.wa}")
    print(f"w: {output_transform.w}")
    print("*********************output_transform***********************")
    print(" ")
    return output_transform

def transform2tcp_position(input_transform: Transform_M,output_position: UR_TCP_Position):
    print("transform2tcp_position()")
    output_position.X = input_transform.x
    output_position.Y = input_transform.y
    output_position.Z = input_transform.z
    print(" ")
    print("*********************input_transform***********************")
    print(f"x: {input_transform.x}")
    print(f"y: {input_transform.y}")
    print(f"z: {input_transform.z}")
    print(f"r11: {input_transform.r11}")
    print(f"r12: {input_transform.r12}")
    print(f"r13: {input_transform.r13}")
    print(f"r21: {input_transform.r21}")
    print(f"r22: {input_transform.r22}")
    print(f"r23: {input_transform.r23}")
    print(f"r31: {input_transform.r31}")
    print(f"r32: {input_transform.r32}")
    print(f"r33: {input_transform.r33}")
    print(f"wn: {input_transform.wn}")
    print(f"wo: {input_transform.wo}")
    print(f"wa: {input_transform.wa}")
    print(f"w: {input_transform.w}")
    print("*********************input_transform***********************")
    print(" ")
    theta = math.acos((input_transform.r11 + input_transform.r22 + input_transform.r33 - 1) / 2.0)
    
    if theta < 0.001 and theta > -0.001:
        output_position.RX = 0.0
        output_position.RY = .0
        output_position.RZ = 0.0
        return output_position

    if theta < (math.pi + 0.1) and theta > (math.pi - 0.1):
        pass

    multi = 1.0 / (2 * math.sin(theta))
    RX = multi * (input_transform.r32 - input_transform.r23)
    RY = multi * (input_transform.r13 - input_transform.r31)
    RZ = multi * (input_transform.r21 - input_transform.r12)

    vector_norm = math.sqrt((RX * RX) + (RY * RY) + (RZ * RZ))

    output_position.RX = theta * (RX / vector_norm)
    output_position.RY = theta * (RY / vector_norm)
    output_position.RZ = theta * (RZ / vector_norm)
    print(" ")
    print("*********************output_position***********************")
    print(f"X: {output_position.X}")
    print(f"Y: {output_position.Y}")
    print(f"Z: {output_position.Z}")
    print(f"rx: {output_position.RX}")
    print(f"ry: {output_position.RY}")
    print(f"rz: {output_position.RZ}")
    print("*********************output_position***********************")
    print(" ")
    return output_position

class JointAngles(Node):
    def __init__(self):
        super().__init__('joint_angles_subscription')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'omni_data2',
            self.msg_callback,
            10)

    def msg_callback(self, msg):
        received_data = msg.data
        x,y,z,rx,ry,rz = received_data
        print("received data: ",received_data)
        received_data_array = np.array(received_data) 
    
        received_data_obj = UR_RPY_Position(
            received_data_array[0],
            received_data_array[1],
            received_data_array[2],
            received_data_array[3],
            received_data_array[4],
            received_data_array[5]
        )
    
        global joint_angles
        if any(math.isnan(val) for val in [x,y,z,rx,ry,rz]):
            self.get_logger().warning("Ignoring received message with NaN values.")
            return
        input_transform = Transform_M()
        RPY_position2transform(received_data_obj,input_transform)
        output_position = UR_TCP_Position(X=0, Y=0, Z=0, RX=0, RY=0, RZ=0)
        transform2tcp_position(input_transform, output_position)
        print("TCP:", output_position.X, output_position.Y, output_position.Z,output_position.RX, output_position.RY, output_position.RZ)
        tcp=np.array([output_position.X, output_position.Y, output_position.Z,output_position.RX, output_position.RY, output_position.RZ])
        
        try:
            print("----------------------------------------------------------------------")
            transform=get_desired_pose_htm(position=np.array(tcp[0:3]),roll=tcp[3],pitch=tcp[4],yaw=tcp[5])
            print("desired pose: ",transform)
            joint_angles = inverse_kinematics(transform, solution_index=5)
            #joint_angles=np.array(joint_angles[:, 1]).flatten().tolist()
            global joint_angles_npArray
            joint_angles_npArray = np.asarray(joint_angles,dtype=np.float32)
            print("joint_angles: ",joint_angles)
            print("----------------------------------------------------------------------")
            return
        
        except ValueError as e:
            print(f"ValueError: {e}")
            return

def main(args=None):
    rclpy.init(args=args)
    joint_angles_subscription = JointAngles()
    spin_thread = threading.Thread(target=lambda: rclpy.spin(joint_angles_subscription), daemon=True)
    spin_thread.start() 
    global joint_angles_npArray
    joint_angles_npArray = np.array([], dtype=np.float32)  #
    try:
        while not gym.query_viewer_has_closed(viewer):
            gym.set_actor_dof_position_targets(env, ur_actor, joint_angles_npArray)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

    except KeyboardInterrupt:
        pass

    joint_angles_subscription.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
