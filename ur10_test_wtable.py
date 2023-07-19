import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import copy

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="UR10 Robot Simulation")
  
args.physics_engine = gymapi.SIM_PHYSX

# set simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 3 
sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)
box_size=0.05
# create ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# set the urdf path and load the asset
asset_root = "/home/son/isaacgym/assets"
urdf_file = "urdf/ur10/ur10_allegro.urdf"

table_dims=gymapi.Vec3(1.0,1.0,0.3)

table_pose=gymapi.Transform()
table_pose.p=gymapi.Vec3(1.0,0.5*table_dims.y+0.001,0.8)

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

table_asset=gym.create_box(sim,table_dims.x,table_dims.y,table_dims.z,asset_options)

asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)

# create an environment
num_envs = 5
env_lower = gymapi.Vec3(0, 0, 0)
env_upper = gymapi.Vec3(3, 3, 3)
env = gym.create_env(sim,env_lower, env_upper, num_envs)

# create a UR10 robot actor in the environment
ur_pose = gymapi.Transform()
ur_pose.p = gymapi.Vec3(0, 0.2, 0)
ur_actor = gym.create_actor(env, asset, ur_pose, "ur10", 0, 1)
table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

# initialize desired joint position
joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


while not gym.query_viewer_has_closed(viewer):

    # update joint targets with desired positions
    for idx in range(6):
        gym.set_actor_dof_position_targets(env, ur_actor, joint_positions)
    # step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
