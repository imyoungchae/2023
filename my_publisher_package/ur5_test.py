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
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
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
asset_options.density = 10.0
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

table_dims=gymapi.Vec3(0.5,0.7,0.5)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(1.0, 1.5, 0)
object_pose = gymapi.Transform()
table_x = table_pose.p.x
table_y = table_pose.p.y
table_z = table_pose.p.z
table_height = table_dims.y

table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)


'''table2_dims=gymapi.Vec3(1.8,1.0,0.7)
table2_pose = gymapi.Transform()
table2_pose.p = gymapi.Vec3(1.2,1.5,-1.0)
'''

table2_dims=gymapi.Vec3(0.9,0.7,0.7)
table2_pose = gymapi.Transform()
table2_pose.p = gymapi.Vec3(1.2,1.5,-1.0)

table_x2 = table2_pose.p.x
table_y2 = table2_pose.p.y
table_z2 = table2_pose.p.z
table2_height = table2_dims.y

table2_asset = gym.create_box(sim, table2_dims.x, table2_dims.y, table2_dims.z, asset_options)

sphere=gym.create_sphere(sim,0.6)
sphere_pose = gymapi.Transform()
sphere_pose.p = gymapi.Vec3(1.2,1.5,-1.0)
# create ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# set the urdf path and load the asset
asset_root = "/home/son/my_ws/src/my_publisher_package/src/isaacgym/assets"
urdf_file = "urdf/ur5/ur5_test3.urdf"

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer,None,gymapi.Vec3(3,3.8,3),gymapi.Vec3(1.1,1.8,-0.15))

# create an environment
num_envs = 1
env_lower = gymapi.Vec3(0, 0, 0)
env_upper = gymapi.Vec3(3, 3, 3)
env = gym.create_env(sim,env_lower, env_upper, num_envs)

# create a UR10 robot actor in the environment
ur_pose = gymapi.Transform()
ur_base_height = 1.2  # Set this value to a suitable height
ur_pose.p = gymapi.Vec3(1.1,0.7,-0.15)
color = gymapi.Vec3(0.4, 0.0, 0.2)  
ur_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
ur_actor = gym.create_actor(env, asset, ur_pose, "ur10", 0, 1)
table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

table_shape_props = gymapi.RigidShapeProperties()
table_shape_props.filter = 1  # Set collision filter for the table
#sphere_actor = gym.create_actor(env, sphere,sphere_pose,  "sphere", 0, 1)
table2_actor = gym.create_actor(env, table2_asset, table2_pose, "table2", 0, 1)
gym.set_rigid_body_color(env, table2_actor, 0,
                         gymapi.MESH_VISUAL_AND_COLLISION,
                         color)
gym.set_asset_rigid_shape_properties(table2_asset, [table_shape_props])

#sphere_actor = gym.create_actor(env, sphere,sphere_pose,  "sphere", 0, 1)

joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'omni_data2',
            self.msg_callback,
            10)

    def msg_callback(self, msg):
        global joint_positions
        received_data = list(msg.data) # Convert received_data to Python List
        joint_positions = np.array(received_data, dtype=np.float32) 
        print("rc data",joint_positions)

def main(args=None):
    rclpy.init(args=args)
    pose_subscriber = PoseSubscriber()
    spin_thread = threading.Thread(target=lambda: rclpy.spin(pose_subscriber), daemon=True)
    spin_thread.start() 
   
    try:
        while not gym.query_viewer_has_closed(viewer):
            gym.set_actor_dof_position_targets(env, ur_actor, joint_positions)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

    except KeyboardInterrupt:
        pass
    pose_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
