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
axes_geom = gymutil.AxesGeometry(0.1)

sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

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

# create ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# set the urdf path and load the asset
asset_root = "/home/son/my_ws/src/my_publisher_package/src/isaacgym/assets"
urdf_file = "urdf/ur10/ur10_test.urdf"

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False

# create an environment
num_envs = 1
env_lower = gymapi.Vec3(0, 0, 0)
env_upper = gymapi.Vec3(3, 3, 3)
env = gym.create_env(sim,env_lower, env_upper, num_envs)

# create a UR10 robot actor in the environment
ur_pose = gymapi.Transform()
ur_base_height = 1.2  # Set this value to a suitable height
ur_pose.p = gymapi.Vec3(1.1,1.8,-0.15)

ur_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
ur_actor = gym.create_actor(env, asset, ur_pose, "ur10", 0, 1)
table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 1)
joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'omni_data',
            self.msg_callback,
            10)

    def msg_callback(self, msg):
        global joint_positions
        received_data = list(msg.data) # Convert received_data to Python List
        print("Received message:", received_data)
        joint_positions = np.array(received_data, dtype=np.float32) 

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
