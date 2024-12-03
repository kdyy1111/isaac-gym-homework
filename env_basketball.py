from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch
import pygame

class Basketball:
    def __init__(self, args):
        self.args = args

        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 60.
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        # Observations:
        # 0:2 - joint DOF positions
        # 2:4 - joint DOF velocities
        # 4:7 - ball position
        # 7:10 - ball linear velocity

        # task-specific parameters
        self.num_obs = 10
        self.reset_dist = 3.0  # when to reset
        self.max_push_effort = 100.0  # the range of force applied to the cartpole
        self.max_episode_length = 500  # maximum episode length
        self.actors_per_env = 2 # arm, ball
        self.dofs_per_env = 2 # joint1, joint2

        # allocate buffers
        self.obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        self.reset_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # initialize envs and state tensors
        self.envs, self.num_dof = self.create_envs()

        # dof_state
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        vec_dof_state_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.args.num_envs, 
                                                                                  self.dofs_per_env, 2)
        self.dof_states = vec_dof_state_tensor
        self.dof_positions = vec_dof_state_tensor[..., 0]
        self.dof_velocities = vec_dof_state_tensor[..., 1]  

        # actor
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.args.num_envs, 
                                                                      self.actors_per_env, 13)
        self.root_states = vec_root_tensor
        self.arm_positions = vec_root_tensor[..., 0, 0:3]
        self.ball_positions = vec_root_tensor[..., 1, 0:3]
        self.ball_orientations = vec_root_tensor[..., 1, 3:7]
        self.ball_linvels = vec_root_tensor[..., 1, 7:10]
        self.ball_angvels = vec_root_tensor[..., 1, 10:13]
        
        # reset
        self.all_actor_indices = torch.arange(self.actors_per_env * self.args.num_envs, 
                                              dtype=torch.int32, device=self.args.sim_device).view(self.args.num_envs, 
                                                                                          self.actors_per_env)
        self.all_arm_indices = self.actors_per_env * torch.arange(self.args.num_envs, 
                                                                   dtype=torch.int32, device=self.args.sim_device)


        # generate viewer for visualisation
        if not self.args.headless:
            self.viewer = self.create_viewer()

        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        self.reset()

        self.initial_dof_states = self.dof_states.clone()
        self.initial_root_states = vec_root_tensor.clone()

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("User Guide")
        
    def handle_keyboard_events(self):
        font = pygame.font.Font(None, 40)
        lines = ["User Guide", "press 'v': rendering stop"]
        self.screen.fill((0, 0, 0))
        y_offset = 100
        line_spacing = 50
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(200, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += line_spacing
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    self.args.headless = not self.args.headless
                    print("############################")
                    print(f"### Render Unabled: {self.args.headless} ###")
                    print("############################")

    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # define environment space
        spacing = 5
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.args.num_envs))

        # create arm asset
        asset_root = 'assets'
        asset_file = 'basketball_arm.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        arm_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_dof = self.gym.get_asset_dof_count(arm_asset)

        # define arm pose
        pose = gymapi.Transform()
        pose.p.z = 3.0
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # define arm dof properties
        dof_props = self.gym.get_asset_dof_properties(arm_asset)
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
        dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0

        # create ball asset
        self.ball_radius = 0.1
        ball_options = gymapi.AssetOptions()
        ball_options.density = 200
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)

        # generate environments
        envs = []
        self.arm_handles = []
        self.ball_handles = []
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add arm in each environment
            arm_handle = self.gym.create_actor(env, arm_asset, pose, "arm", i, 0, 0)
            self.gym.set_actor_dof_properties(env, arm_handle, dof_props)
            self.arm_handles.append(arm_handle)

            # add ball in each environment
            ball_pose = gymapi.Transform()
            ball_pose.p.x = 0
            ball_pose.p.y = 0.7
            ball_pose.p.z = 3.0
            ball_handle = self.gym.create_actor(env, ball_asset, ball_pose, "ball", i, 0, 0)
            self.ball_handles.append(ball_handle)

            # pretty colors
            self.gym.set_rigid_body_color(env, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.99, 0.66, 0.25))

            envs.append(env)
        return envs, num_dof

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer

    def get_obs(self, env_ids=None):
        # get state observation from each environment id
        if env_ids is None:
            env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        actuated_dof_indices = torch.tensor([0, 1], device=self.args.sim_device)

        self.obs_buf[..., 0:2] = self.dof_positions[..., actuated_dof_indices]
        self.obs_buf[..., 2:4] = self.dof_velocities[..., actuated_dof_indices]
        self.obs_buf[..., 4:7] = self.ball_positions
        self.obs_buf[..., 7:10] = self.ball_linvels
        
    def get_reward(self):
        self.reward_buf[:], self.reset_buf[:] = compute_arm_reward(
            self.ball_positions,
            self.ball_linvels,
            self.ball_radius,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return

        positions = torch.zeros((len(env_ids), self.num_dof), device=self.args.sim_device)
        velocities = torch.zeros((len(env_ids), self.num_dof), device=self.args.sim_device)

        self.dof_positions[env_ids, :] = positions[:]
        self.dof_velocities[env_ids, :] = velocities[:]

        self.ball_positions[env_ids, 0] = 0
        self.ball_positions[env_ids, 2] = 3
        self.ball_positions[env_ids, 1] = 0.7
        self.ball_orientations[env_ids, 0:3] = 0
        self.ball_orientations[env_ids, 3] = 1
        self.ball_linvels[env_ids, 0] = 0.0
        self.ball_linvels[env_ids, 2] = 0.0
        self.ball_linvels[env_ids, 1] = 0.0
        self.ball_angvels[env_ids] = 0

        # reset root state for arms and balls in selected envs
        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     self.root_tensor, 
                                                     gymtorch.unwrap_tensor(actor_indices), 
                                                     len(actor_indices))
        
        # reset dof_states in selected envs
        arm_indices = self.all_arm_indices[env_ids].flatten()
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim, 
                                              self.dof_state_tensor, 
                                              gymtorch.unwrap_tensor(arm_indices), 
                                              len(arm_indices))

        # clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # refresh new observation after reset
        self.get_obs()

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def exit(self):
        # close the simulator in a graceful way
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def step(self, actions):
        # apply action
        actions_tensor = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        
        ###################################################
        actions_tensor.fill_(100)
        ###################################################

        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        self.handle_keyboard_events()

        # simulate and render
        self.simulate()
        if not self.args.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1
        self.get_obs()
        self.get_reward()


@torch.jit.script
def compute_arm_reward(ball_positions, ball_velocities, ball_radius, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
 
    ###################################################
    goal_position = torch.tensor([0.0, 3.0, 3.0], device="cuda")
    goal_threshold = 0.3
    ball_position = ball_positions

    distance_to_goal = torch.norm(ball_position - goal_position, dim=-1)

    reward = torch.where(distance_to_goal < goal_threshold, torch.tensor(100.0), torch.tensor(0.0))
    reward += torch.where(ball_positions[..., 1] < 0.6, torch.ones_like(reward) * -20.0, reward)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(ball_positions[..., 1] < 0.6, torch.ones_like(reset_buf), reset)
    reset = torch.where(distance_to_goal < goal_threshold - 0.1, torch.ones_like(reset_buf), reset)
    ###################################################
    
    return reward, reset    

