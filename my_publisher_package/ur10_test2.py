#!/usr/bin/env python3
import math
import numpy as np
from isaacgym import gymapi
import isaacgym
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
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

global mat
mat=np.matrix

gym = gymapi.acquire_gym()
axes_geom = gymutil.AxesGeometry(0.1)
joint_angles = np.zeros(6, dtype=np.float32)
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

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

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer,None,gymapi.Vec3(3,3.8,3),gymapi.Vec3(1.1,1.8,-0.15))
#gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

if viewer is None:
    print("*** Failed to create viewer")
    quit()

#gym.set_camera_transform(camera_handle, env, camera_transform)
ur_pose = gymapi.Transform()
ur_base_height = 1.2 
ur_pose.p = gymapi.Vec3(1.1,1.8,-0.15)

ur_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
ur_actor = gym.create_actor(env, asset, ur_pose, "ur10", 0, 1)

table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 1)
angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

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


def forward_kinematic_solution(DH_matrix, edges=np.matrix([[0], [0], [0], [0], [0], [0]])):
    t01 = mat_transtorm_DH(DH_matrix, 1, edges)
    t12 = mat_transtorm_DH(DH_matrix, 2, edges)
    t23 = mat_transtorm_DH(DH_matrix, 3, edges)
    t34 = mat_transtorm_DH(DH_matrix, 4, edges)
    t45 = mat_transtorm_DH(DH_matrix, 5, edges)
    t56 = mat_transtorm_DH(DH_matrix, 6, edges)
    answer = t01 * t12 * t23 * t34 * t45 * t56
    return answer


def inverse_kinematic_solution(DH_matrix, transform_matrix,):
    theta = np.matrix(np.zeros((6, 8)))
    # theta 1
    T06 = transform_matrix
    P05 = T06 * np.matrix([[0], [0], [-DH_matrix[5, 2]], [1]])
    psi = atan2(P05[1], P05[0])
    phi = acos((DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2]) / sqrt(P05[0] ** 2 + P05[1] ** 2))
    theta[0, 0:4] = psi + phi + pi / 2
    theta[0, 4:8] = psi - phi + pi / 2

    # theta 5
    for i in {0, 4}:
        th5cos = (T06[0, 3] * sin(theta[0, i]) - T06[1, 3] * cos(theta[0, i]) - (
                DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2])) / DH_matrix[5, 2]
        if 1 >= th5cos >= -1:
            th5 = acos(th5cos)
        else:
            th5 = 0
        theta[4, i:i + 2] = th5
        theta[4, i + 2:i + 4] = -th5
    # theta 6
    for i in {0, 2, 4, 6}:
        # if sin(theta[4, i]) == 0:
        #     theta[5, i:i + 1] = 0 # any angle
        #     break
        T60 = linalg.inv(T06)
        th = atan2((-T60[1, 0] * sin(theta[0, i]) + T60[1, 1] * cos(theta[0, i])),
                   (T60[0, 0] * sin(theta[0, i]) - T60[0, 1] * cos(theta[0, i])))
        theta[5, i:i + 2] = th

    # theta 3
    for i in {0, 2, 4, 6}:
        T01 = mat_transtorm_DH(DH_matrix, 1, theta[:, i])
        T45 = mat_transtorm_DH(DH_matrix, 5, theta[:, i])
        T56 = mat_transtorm_DH(DH_matrix, 6, theta[:, i])
        T14 = linalg.inv(T01) * T06 * linalg.inv(T45 * T56)
        P13 = T14 * np.matrix([[0], [-DH_matrix[3, 2]], [0], [1]])
        costh3 = ((P13[0] ** 2 + P13[1] ** 2 - DH_matrix[1, 0] ** 2 - DH_matrix[2, 0] ** 2) /
                  (2 * DH_matrix[1, 0] * DH_matrix[2, 0]))
        if 1 >= costh3 >= -1:
            th3 = acos(costh3)
        else:
            th3 = 0
        theta[2, i] = th3
        theta[2, i + 1] = -th3

    # theta 2,4
    for i in range(8):
        T01 = mat_transtorm_DH(DH_matrix, 1, theta[:, i])
        T45 = mat_transtorm_DH(DH_matrix, 5, theta[:, i])
        T56 = mat_transtorm_DH(DH_matrix, 6, theta[:, i])
        T14 = linalg.inv(T01) * T06 * linalg.inv(T45 * T56)
        P13 = T14 * np.matrix([[0], [-DH_matrix[3, 2]], [0], [1]])

        theta[1, i] = atan2(-P13[1], -P13[0]) - asin(
            -DH_matrix[2, 0] * sin(theta[2, i]) / sqrt(P13[0] ** 2 + P13[1] ** 2)
        )
        T32 = linalg.inv(mat_transtorm_DH(DH_matrix, 3, theta[:, i]))
        T21 = linalg.inv(mat_transtorm_DH(DH_matrix, 2, theta[:, i]))
        T34 = T32 * T21 * T14
        theta[3, i] = atan2(T34[1, 0], T34[0, 0])
    return theta

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
        x, y, z, rx, ry, rz = received_data
        global angles
        if any(math.isnan(val) for val in [x, y, z, rx, ry, rz]):
            self.get_logger().warning("Ignoring received message with NaN values.")
            return
        ed = received_data
        transform = forward_kinematic_solution(DH_matrix_UR10, ed)
        IKS = inverse_kinematic_solution(DH_matrix_UR10, transform)
        angles = np.array(IKS[:, 5]).flatten().tolist()
        print("angles=",angles)

def main(args=None):
    rclpy.init(args=args)
    joint_angles_subscription = JointAngles()
    spin_thread = threading.Thread(target=lambda: rclpy.spin(joint_angles_subscription), daemon=True)
    spin_thread.start() 

    try:
        while not gym.query_viewer_has_closed(viewer):

            gym.set_actor_dof_position_targets(env, ur_actor, angles)
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
