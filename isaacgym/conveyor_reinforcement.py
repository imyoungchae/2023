import numpy as np
import os
import torch
import cv2
from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]


        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_goal": self.cfg["env"]["goalRewardScale"],
            "r_endpenalty": self.cfg["env"]["endpenalty"],
            "r_distance_reward":self.cfg["env"]["tabledistRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensionsenv
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 20 if self.control_type == "osc" else 30
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 6 if self.control_type == "osc" else 7

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeA_id = None     

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self.franka_dof_pos = None  # Joint positions           (n_envs, n_dof)
        self.franka_dof_vel = None                     # Joint velocities          (n_envs, n_dof)
        self.conveyor_dof_vel=None
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        self.conveyor_target_state=None
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [-0.4745,-1.4679, 1.9761, -2.0973,-1.6185,-2.3006,0.5000, 0.5000,-0.5000,-0.5000], device=self.device
        )
        self.conveyor_default_vel=to_torch([0.6], device=self.device)
        self.conveyor_default_pos=to_torch([0],device=self.device)

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:6].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "/home/son/isaacgym/assets" 
        franka_asset_file = "urdf/ur10_sy/urdf/ur10_gripper2.urdf"
        conveyor_asset_file = "urdf/conveyor_urdf.urdf"
        redbox_asset_file = "urdf/box4cube.urdf"
        greenbox_asset_file = "urdf/box4cube.urdf"
        bluebox_asset_file = "urdf/box4cube.urdf"



        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            conveyor_asset_file = self.cfg["env"]["asset"].get("assetFileNameConveyor", conveyor_asset_file)
            redbox_asset_file = self.cfg["env"]["asset"].get("redbox", redbox_asset_file)
            greenbox_asset_file = self.cfg["env"]["asset"].get("greenbox", greenbox_asset_file)
            bluebox_asset_file = self.cfg["env"]["asset"].get("bluebox", bluebox_asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True

        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        conveyor_asset=self.gym.load_asset(self.sim, asset_root, conveyor_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 500., 500.,500., 500.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True

        self.table_stand_size=0.3
        # Create table stand asset
        table_stand_height = 0.1

        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        #table_stand_asset = self.gym.create_box(self.sim,*([self.table_stand_size] * 3), table_opts)
        table_stand_asset = self.gym.load_asset(self.sim, asset_root, redbox_asset_file, table_opts)
        table_stand_asset2 = self.gym.load_asset(self.sim, asset_root, bluebox_asset_file, table_opts)
        table_stand_asset3 = self.gym.load_asset(self.sim, asset_root, greenbox_asset_file, table_opts)

        robot_stand_height = 0.1
        robot_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        robot_stand_opts = gymapi.AssetOptions()
        robot_stand_opts.fix_base_link = True
        robot_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.4, robot_stand_height], table_opts)

        box_stand_asset = self.gym.create_box(self.sim, *[0.6, 1.1, robot_stand_height], table_opts)  

        self.cubeA_size = 0.055
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.1,0.7,0.1)
        table_stand_color=gymapi.Vec3(0.973,0.244,0.244)
        table_stand_color2=gymapi.Vec3(0.435,0.811,0.266)
        table_stand_color3=gymapi.Vec3(0.089,0.309,0.548)
        robot_stand_color=gymapi.Vec3(0.1,0.1,0.1)
        asset_options = gymapi.AssetOptions()
  
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        self.num_conveyor_bodies = self.gym.get_asset_rigid_body_count(conveyor_asset)
        self.num_conveyor_dofs = self.gym.get_asset_dof_count(conveyor_asset)

        print("num conveyor bodies: ", self.num_conveyor_bodies)
        print("num conveyor dofs: ", self.num_conveyor_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 400.0
                franka_dof_props['damping'][i] = 80.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200
 
        conveyor_dof_props = self.gym.get_asset_dof_properties(conveyor_asset)
        conveyor_dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL) # Or DOF_MODE_POS depending 
        conveyor_dof_props["stiffness"] = (5000.0)
        conveyor_dof_props["damping"] = (200.0)

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_pos_box =[-0.8, 0.35, 1.0 ]
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos_box)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_stand_start_pose2 = gymapi.Transform()
        table_stand_pos_box2 = [-0.8, 0.7, 1.0 ]
        table_stand_start_pose2.p = gymapi.Vec3(*table_stand_pos_box2)
        table_stand_start_pose2.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_stand_start_pose3 = gymapi.Transform()
        table_stand_pos_box3 = [-0.8, 1.05, 1.0 ]
        table_stand_start_pose3.p = gymapi.Vec3(*table_stand_pos_box3)
        table_stand_start_pose3.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table stand
        robot_stand_start_pose = gymapi.Transform()
        robot_stand_start_pose.p = gymapi.Vec3(*robot_stand_pos)
        robot_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        box_stand_start_pose = gymapi.Transform()
        box_stand_pos = [-0.8, 0.7, 0.87 ]
        box_stand_start_pose.p = gymapi.Vec3(*box_stand_pos)
        box_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        conveyor_start_pose=gymapi.Transform()
        conveyor_start_pose.p = gymapi.Vec3(*table_pos)
        conveyor_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)

        max_agg_bodies = num_franka_bodies + 8 # 1 for table, table stand, cubeA
        max_agg_shapes = num_franka_shapes + 8    # 1 for table, table stand, cubeA'

        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 640

        self.frankas = []
        self.envs = []
        self.conveyors=[]
        # Create environments
        for i in range(self.num_envs):
            # create env instance

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
            
            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            conveyor_actor = self.gym.create_actor(env_ptr, conveyor_asset, conveyor_start_pose, "conveyor", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, conveyor_actor, conveyor_dof_props)

            target_dof_actor = self.gym.find_actor_dof_handle(env_ptr,conveyor_actor, 'slider_to_cart')
            self.gym.set_dof_target_velocity(env_ptr,target_dof_actor,0.6)

            self.table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)
            self.table_stand_actor2 = self.gym.create_actor(env_ptr, table_stand_asset2, table_stand_start_pose2, "table_stand2",
                                                      i, 1, 0)
            self.table_stand_actor3 = self.gym.create_actor(env_ptr, table_stand_asset3, table_stand_start_pose3, "table_stand3",
                                                      i, 1, 0)
            self.robot_stand_actor = self.gym.create_actor(env_ptr, robot_stand_asset, robot_stand_start_pose, "robot_stand",
                                                      i, 1, 0)
            self.box_stand_actor = self.gym.create_actor(env_ptr, box_stand_asset, box_stand_start_pose, "box_stand",
                                                      i, 1, 0)
            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self.table_stand_actor, 0, gymapi.MESH_VISUAL, table_stand_color)
            self.gym.set_rigid_body_color(env_ptr, self.table_stand_actor2, 0, gymapi.MESH_VISUAL, table_stand_color2)
            self.gym.set_rigid_body_color(env_ptr, self.table_stand_actor3, 0, gymapi.MESH_VISUAL, table_stand_color3)
            self.gym.set_rigid_body_color(env_ptr, self.robot_stand_actor, 0, gymapi.MESH_VISUAL, robot_stand_color)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.conveyors.append(conveyor_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        conveyor_handle=0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "ee_link"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "ur10_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "ur10_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "ur_grip_site"),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "target_dof_actor":self.gym.find_actor_rigid_body_handle(env_ptr,conveyor_handle, 'slider_to_cart'),
            "table_handle":self.gym.find_actor_rigid_body_handle(self.envs[0], self.table_stand_actor2, "table_box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)

        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
  
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self.franka_dof_state = self._dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self._conveyor_dof_state= self._dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self._conveyor_dof_pos = self._conveyor_dof_state[..., 0]
        self._conveyor_dof_vel = self._conveyor_dof_state[..., 1]

        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        self.conveyor_target_state= self._rigid_body_state[:, self.handles["target_dof_actor"], :]
 
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :6]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._table_stand_actor_state = self._root_state[:, self.table_stand_actor2, :]

        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "table_stand_size": torch.ones_like(self._eef_state[:, 0]) * self.table_stand_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, 11), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :6]
        self._gripper_control = self._pos_control[:, 6:10]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 8, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self.franka_dof_pos[:, :],
            "q_gripper": self.franka_dof_vel[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],


            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_vel": self._cubeA_state[:, 7:10],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "table_box_pos": self._table_stand_actor_state[:, :3],
            "table_box_quat": self._table_stand_actor_state[:, 3:7],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "eef_pos", "eef_quat", "cubeA_vel","table_box_pos"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self._reset_init_cube_state(cube='A', env_ids=env_ids)
        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        
        # Reset agent
        reset_noise = torch.rand((len(env_ids), 10), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -4:] = self.franka_default_dof_pos[-4:]

        self._conveyor_dof_pos[env_ids, :]=self.conveyor_default_pos[:]

        # Reset the internal obs accordingly
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :10] = pos
        self._effort_control[env_ids, :self.num_franka_dofs] = torch.zeros_like(pos)
        #elf.conveyor_default_vel[env_ids, :] = torch.zeros_like(self.conveyor_default_vel)
        self._pos_control[env_ids, -1] = self.conveyor_default_vel

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()


 
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected

        this_cube_state_all = self._init_cubeA_state
        cube_heights = self.states["cubeA_size"]
 
        # Sampling is "centered" around middle of table
        centered_cube_x_state = torch.tensor(self._table_surface_pos[0], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        sampled_cube_state[:, 1]= - 1.0
        sampled_cube_state[:, :1] = centered_cube_x_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 1, device=self.device) - 0.5)
   
        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])
  
        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self.franka_dof_pos[:, :6], self.franka_dof_vel[:, :6]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 6:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(6, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:6].unsqueeze(0), self._franka_effort_limits[:6].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
     
        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :6], self.actions[:, -4:]

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        u_fingers = torch.zeros_like(self._gripper_control, device=self.device)
        u_gripper_expanded = u_gripper.expand_as(self._gripper_control).to(self.device)

        u_fingers[:, :2] = torch.where(u_gripper_expanded[:, :2] >= 0.0, torch.tensor([[0.2, 0.1]], device=self.device),
                                        torch.tensor([[-0.2, -0.1]], device=self.device))

        u_fingers[:, 2:] = torch.where(u_gripper_expanded[:, 2:] >= 0.0, torch.tensor([[0.9, 0.9]], device=self.device),
                                        torch.tensor([[-0.9, -0.9]], device=self.device))

        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.5,0.8,2.8)
            cam_target = gymapi.Vec3(0,0,0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
                
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                
                for pos, rot in zip((eef_pos, cubeA_pos), (eef_rot, cubeA_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
        
        '''rgb_filename = "/home/son/isaacgym/python/examples/multiple_camera_images/rgb_env7%d_cam.png"
        for i in range(20):
            self.gym.write_viewer_image_to_file(self.viewer, rgb_filename %i)'''
###################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(

    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    target_height = reward_settings["table_height"]+states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    table_stand_size=states["table_stand_size"]


    # Distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
    #print("dist reward:",dist_reward[0])

    # Reward for lifting cubeA
    cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
    cubeA_lifted = ((cubeA_height - cubeA_size) > 0.075)
    #cubeA_lifted = (cubeA_height - cubeA_size) > 0.075
    #print("cube A height:",cubeA_height[0]-cubeA_size[0])
    lift_reward = cubeA_lifted

    cubeA_to_table_box_distance = torch.norm(states["table_box_pos"][:,:2] - states["cubeA_pos"][:,:2] , dim=-1)

    distance_reward = (1 - torch.tanh(10.0 * (cubeA_to_table_box_distance)))*cubeA_lifted
    #distance_reward= torch.where((cubeA_to_table_box_distance < 0.4)&(dist_reward>0.4)&cubeA_lifted,(1 - torch.tanh(10.0 * cubeA_to_table_box_distance))*((1-cubeA_lifted)+1.5),(1 - torch.tanh(10.0 * (cubeA_to_table_box_distance)))*cubeA_lifted)
    #print("cubeA_to_table_box_distance ",cubeA_to_table_box_distance[0])

    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    cubeA_align_cubeB = (torch.norm(states["cubeA_pos"][:, :2] - states["table_box_pos"][:, :2], dim=-1) < 0.1)
    #print("cubeA_align_cubeB ",cubeA_align_cubeB[0])
    #print("cubeA_height",cubeA_height[0])
    #print("target_height",target_height[0])
    #print("torch.abs(cubeA_height - target_height)",torch.abs(cubeA_height[0] - target_height[0]))
    gripper_away_from_cubeA = (d > 0.04)
    #print("gripper_away_from_cubeA  ",gripper_away_from_cubeA[0])
    goal_reward = cubeA_align_cubeB  & gripper_away_from_cubeA
    #rint("goal_reward=",goal_reward[0])
    #print("----------------------------------------------------------------")
    # reset
    reset_condition = (states["cubeA_pos"][:, 1] > 1.1)
    # end penalty
    end_penalty = (states["cubeA_pos"][:, 1] > 1.1)

    rewards = torch.where(
        goal_reward,
        reward_settings["r_goal"] * goal_reward,
        reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings["r_distance_reward"]*distance_reward - reward_settings["r_endpenalty"] * end_penalty)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (reset_condition>0) | (goal_reward > 0), torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf
