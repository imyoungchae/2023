from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
import numpy as np
import torch

def quat_axis(q, axis=1):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def cube_grasping_yaw(q, corners):
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 2], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw

    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w],dim=-1)

    angle = torch.tensor([90 * (3.14159265 / 180)]).cuda()
    q_y = torch.cat((torch.cos(angle / 2), torch.zeros(1).cuda(), torch.sin(angle / 2), torch.zeros(1).cuda()))
    N_yaw_quats=yaw_quats.shape[0]
    q_y=q_y.repeat(N_yaw_quats,1)
    yaw_quats= quat_mul(q_y, yaw_quats)

    return yaw_quats

def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    return u

# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for ur5. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 5, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="ur5 Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.055

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")  

asset_root = "/home/son/isaacgym/assets" 

# create table asset
table_dims = gymapi.Vec3(0.6, 1.2 ,0.6)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)


table2_dims = gymapi.Vec3(0.4, 0.4 ,0.4)
table2_asset = gym.create_box(sim,table2_dims.x, table2_dims.y, table2_dims.z, asset_options)

# create box asset
box_size = 0.05
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

ur5_asset_file = "urdf/ur10_sy/urdf/ur10_gripper.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments =False
ur5_asset = gym.load_asset(sim, asset_root, ur5_asset_file, asset_options)

# configure ur5 dofs
ur5_dof_props = gym.get_asset_dof_properties(ur5_asset)
ur5_lower_limits = ur5_dof_props["lower"]
ur5_upper_limits = ur5_dof_props["upper"]
ur5_ranges = ur5_upper_limits - ur5_lower_limits
ur5_mids = 0.3 * (ur5_upper_limits + ur5_lower_limits)

# use position drive for all dofs
if controller == "ik":
    ur5_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_POS)
    ur5_dof_props["stiffness"][:6].fill(400.0)
    ur5_dof_props["damping"][:6].fill(40.0)
else:       # osc
    ur5_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_EFFORT)
    ur5_dof_props["stiffness"][:6].fill(0.0)
    ur5_dof_props["damping"][:6].fill(0.0)
# grippers
ur5_dof_props["driveMode"][6:].fill(gymapi.DOF_MODE_POS)
ur5_dof_props["stiffness"][6:].fill(400.0)
ur5_dof_props["damping"][6:].fill(40.0)

# default dof states and position targets
ur5_num_dofs = gym.get_asset_dof_count(ur5_asset)
default_dof_pos = np.zeros(ur5_num_dofs, dtype=np.float32)
default_dof_pos[:6] = ur5_mids[:6]

# grippers open
default_dof_pos[6:] = ur5_upper_limits[6:]

default_dof_state = np.zeros(ur5_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
ur5_link_dict = gym.get_asset_rigid_body_dict(ur5_asset)
ur5_hand_index = ur5_link_dict["hand_link"]
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

ur5_pose = gymapi.Transform()
ur5_pose.p = gymapi.Vec3(-0.5, 0, 0.5)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

table2_pose = gymapi.Transform()

box_pose = gymapi.Transform()

envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []
table2_idxs=[]
# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    table_color = gymapi.Vec3(0.1, 0.1, 0.1)  
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

    table2_pose.p.x = table_pose.p.x-1
    table2_pose.p.y = table_pose.p.y
    table2_pose.p.z=table_pose.p.z

    table2_handle = gym.create_actor(env, table2_asset, table2_pose, "table2", i, 0)

    # add box
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)

    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    table2_idx=gym.get_actor_rigid_body_index(env,table2_handle,0,gymapi.DOMAIN_SIM)
    table2_idxs.append(table2_idx)

    ur5_handle = gym.create_actor(env, ur5_asset, ur5_pose, "ur5", i,1)

    # set dof properties
    gym.set_actor_dof_properties(env, ur5_handle, ur5_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, ur5_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, ur5_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, ur5_handle, "hand_link")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.z, hand_pose.p.y])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.z, hand_pose.r.y, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env,ur5_handle, "hand_link", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([0.0, 1.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

table2_half_size = 0.5 * 0.4
table2_coord = torch.Tensor([table2_half_size, table2_half_size,table2_half_size])
table2_coords = torch.stack(num_envs * [table2_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0,0,-1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base ur5, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "ur5")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to ur5 hand
j_eef = jacobian[:, ur5_hand_index - 1, :, :6]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "ur5")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :6, :6]       

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 10, 1)
dof_vel = dof_states[:, 1].view(num_envs, 10, 1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]
    print("box pos:",box_pos)
    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]
    print("hand pos:",hand_pos)
    to_box = box_pos - hand_pos
    to2_box=hand_pos[:,2]-box_pos[:,2]
    print("2_box",to2_box)
    print("----------------------")
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dist=box_dist
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.09 if controller == "ik" else 0.12

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 8] + dof_pos[:, 9] 
    gripped = (gripper_sep < 0.5) & (box_dist < grasp_offset + 0.5*box_size)

    yaw_q = cube_grasping_yaw(box_rot, corners)

    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot,0)

    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)
 
    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    
    fpose=torch.Tensor([0.2,0.2,0.8]).to(device)
    fori=torch.Tensor([0.5,0,0.0,0.0]).to(device)

    hand_restart = (hand_restart & (init_dist > 0.1)).squeeze(-1)
 
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset*3 )).squeeze(-1)
    grasp_pos = box_pos.clone()


    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2]+ grasp_offset*0.8, box_pos[:, 2] + grasp_offset * 2.1)

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, fpose, grasp_pos)
    goal_rot = torch.where(return_to_start,fori, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    pos_action[:, :6] = dof_pos.squeeze(-1)[:, :6] + control_ik(dpose)

    # gripper actions depend on distance between hand and box
    close_gripper = (box_dist < grasp_offset+0.1) | gripped
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    hand_restart = hand_restart | (box_pos[:, 2] > 1.0)
    keep_going = torch.logical_not(hand_restart)

    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[-0.2,-0.1,0.2,0.1]] * num_envs).to(device),torch.Tensor([[0.5,0.5,-0.5,-0.5]] * num_envs).to(device))
    pos_action[:, 6:10] = grip_acts
    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
  
