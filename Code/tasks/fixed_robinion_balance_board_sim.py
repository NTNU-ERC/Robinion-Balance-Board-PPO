# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from distutils.log import error
from logging import root
from operator import index
import gym
import numpy as np
import os
import torch
import math
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

from isaacgymenvs.utils import balance_board_utils
from enum import IntEnum, auto

class ControlMode(IntEnum):
    free = auto()
    parallel_pitch = auto()
    inverse_kinematics = auto()

class ControlledJoints(IntEnum):
    all_joints = auto()
    legs_only = auto()
    legs_shoulder_roll = auto()
    legs_shoulder_roll_pitch = auto()

class FixedRobinionBalanceBoardSim(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.debug_mode = self.cfg["debug_mode"]
        self.tolerance_disturbance_mode = self.cfg["tolerance_disturbance_mode"]
        self.board_pitch_mode = self.cfg["board_pitch_mode"]

        self.load_cfg_params()

        _ctrl_mode = self.cfg["robot"]["control_mode"]
        _controlled_joints = self.cfg["robot"]["controlled_joints"]

        if _ctrl_mode == "free" and _controlled_joints == "all_joints":
            self.p_control_mode = ControlMode.free
            self.p_controlled_joints = ControlledJoints.all_joints

        elif _ctrl_mode == "free" and _controlled_joints == "legs_only":
            self.p_control_mode = ControlMode.free
            self.p_controlled_joints = ControlledJoints.legs_only

        elif _ctrl_mode == "parallel_pitch" and _controlled_joints == "all_joints":
            self.p_control_mode = ControlMode.parallel_pitch
            self.p_controlled_joints = ControlledJoints.all_joints

        elif _ctrl_mode == "parallel_pitch" and _controlled_joints == "legs_only":
            self.p_control_mode = ControlMode.parallel_pitch
            self.p_controlled_joints = ControlledJoints.legs_only

        elif _ctrl_mode == "parallel_pitch" and _controlled_joints == "legs_shoulder_roll":
            self.p_control_mode = ControlMode.parallel_pitch
            self.p_controlled_joints = ControlledJoints.legs_shoulder_roll

        elif _ctrl_mode == "parallel_pitch" and _controlled_joints == "legs_shoulder_roll_pitch":
            self.p_control_mode = ControlMode.parallel_pitch
            self.p_controlled_joints = ControlledJoints.legs_shoulder_roll_pitch

        elif _ctrl_mode == "inverse_kinematics":
            raise NotImplementedError
            self.p_control_mode = ControlMode.parallel_pitch

        self.mode_joint_name = self.p_control_mode.name + "_" + self.p_controlled_joints.name
        self.cfg["env"]["numObservations"] = 3 * len(self.mode_joints_dict[self.mode_joint_name + "_dof"]) + 24
        self.cfg["env"]["numActions"] = len(self.mode_joints_dict[self.mode_joint_name + "_dof"])
        if self.p_control_mode == ControlMode.parallel_pitch:
            self.cfg["env"]["numActions"] -= 4
        
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        self.extras = {}
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"bongo_board": torch_zeros(), "upright": torch_zeros(), "bongo_vel" : torch_zeros(), "l_foot_position" : torch_zeros()}

        self.allocate_util_tensors()

        if self.viewer != None:
            cam_pos = gymapi.Vec3(5.0, 5.5, 0.85)
            cam_target = gymapi.Vec3(10.0, 10.0, 0.35)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def load_cfg_params(self):
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.p_dt = self.cfg["sim"]["dt"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.p_robot_reset_height_threshold = self.cfg["env"]["reset_height_threshold"]

        self.p_min_roller_height = self.cfg["env"]["reset"]["min_height"]
        self.p_max_roller_height = self.cfg["env"]["reset"]["max_height"]
        self.p_reset_range_height = self.p_max_roller_height - self.p_min_roller_height
        self.p_robot_roller_reset_offset_height = self.cfg["env"]["reset"]["robot_roller_reset_offset"]

        self.p_controlled_joints_stiffness = self.cfg["robot"]["joint_property"]["controlled_joints"]["stiffness"]
        self.p_controlled_joints_damping = self.cfg["robot"]["joint_property"]["controlled_joints"]["damping"]
        self.p_fixed_joints_stiffness = self.cfg["robot"]["joint_property"]["fixed_joints"]["stiffness"]
        self.p_fixed_joints_damping = self.cfg["robot"]["joint_property"]["fixed_joints"]["damping"]
        
        self.free_all_joints_dof = self.cfg["robot"]["joints_list"]["free"]["all_joints"]
        self.free_legs_only_dof = self.cfg["robot"]["joints_list"]["free"]["legs_only"]
        self.parallel_pitch_all_joints_dof = self.cfg["robot"]["joints_list"]["parallel_pitch"]["all_joints"]
        self.parallel_pitch_legs_only_dof = self.cfg["robot"]["joints_list"]["parallel_pitch"]["legs_only"]
        self.parallel_pitch_legs_shoulder_roll_dof = self.cfg["robot"]["joints_list"]["parallel_pitch"]["legs_shoulder_roll"]
        self.parallel_pitch_legs_shoulder_roll_pitch_dof = self.cfg["robot"]["joints_list"]["parallel_pitch"]["legs_shoulder_roll_pitch"]
        self.mode_joints_dict = {"free_all_joints_dof": self.free_all_joints_dof,
                                        "free_legs_only_dof": self.free_legs_only_dof,
                                        "parallel_pitch_all_joints_dof": self.parallel_pitch_all_joints_dof,
                                        "parallel_pitch_legs_only_dof": self.parallel_pitch_legs_only_dof,
                                        "parallel_pitch_legs_shoulder_roll_dof": self.parallel_pitch_legs_shoulder_roll_dof,
                                        "parallel_pitch_legs_shoulder_roll_pitch_dof": self.parallel_pitch_legs_shoulder_roll_pitch_dof}

        self.p_bongo_board_reward_weight = self.cfg["env"]["reward_weight"]

    def allocate_util_tensors(self):
        self.roller_id = torch.tensor([2*x for x in range(self.num_envs)], 
            device=self.device, dtype=torch.long)
        self.robot_id = self.roller_id + 1

        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) # Simulation format. shape : num_actors, 13 
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor) # PyTorch, Torch Tensor.        
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_linear_vels = self.root_tensor[:, 7:10]
        self.root_angular_vels = self.root_tensor[:, 10:13]

        self._dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # shape : num_envs * num_dofs, 2
        self.dof_state_tensor = gymtorch.wrap_tensor(self._dof_state_tensor)
        self.dof_pos = self.dof_state_tensor.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state_tensor.view(self.num_envs, self.num_dof, 2)[..., 1] 

        self._rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # shape: num_envs, num_rigid_bodies, 13
        self.rigid_body_state = gymtorch.wrap_tensor(self._rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_state.shape[1]
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.reset_root_tensor = self.root_tensor.clone() 
        self.reset_root_tensor[:, 7:13] = 0.0
        self.reset_dof_pos = self.dof_pos.clone()
        self.reset_dof_vel = self.dof_vel.clone()

        self.pd_targets_tensor = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float32)
        self.default_pose_tensor = torch.zeros_like(self.pd_targets_tensor)

        if self.p_control_mode == ControlMode.parallel_pitch:
            self.parallel_pitch_weight_tensor = torch.ones(len(self.controlled_joints_index), device=self.device, dtype=torch.float32)
            self.parallel_pitch_weight_tensor[[self.l_knee_pitch_index_index, self.r_knee_pitch_index_index]] = -2.0 #knee angle will be twice than the hip angle

        # Unit vector pointing up, along the Z axis.
        self.z_basis_vec = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.z_basis_vec[:, 2] = 1.0
        
        self.log_reward_tensor = torch.zeros((4, self.num_envs), device=self.device, dtype=torch.float32)

        #Compute the average disturbance the robot can tolerate 
        if self.tolerance_disturbance_mode:
            self.forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            self.tolerance_count = 0
            self.force_magnitude = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self.total_force = 0
            self.fall_time = 0

        #Compute the avarage board angle
        if self.board_pitch_mode:
            self.board_pitch_ang_vel_sum = torch.zeros(1, device = self.device, dtype=torch.float32)
            self.average_board_pitch_ang_vel = torch.zeros(1, device = self.device, dtype=torch.float32)
            self.step_count = torch.zeros(1, device = self.device, dtype=torch.float32)

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        # If randomizing, apply once immediately on startup before the first sim step
        if self.randomize:
           self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["plane"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.gym.add_ground(self.sim, plane_params)

    def load_roller_asset(self):
        assets_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_roller_path = "urdf/bongo_board/roller.urdf"
        asset_roller_options = gymapi.AssetOptions()
        asset_roller_options.fix_base_link = False
        asset_roller_options.replace_cylinder_with_capsule = True
        
        self.asset_roller = self.gym.load_asset(self.sim, assets_root_dir, asset_roller_path, asset_roller_options)
        if self.debug_mode:
            balance_board_utils.print_asset_info(self.asset_roller,"Roller", self.gym)
            input("Here is the info of Roller, press any key to continue")

        self.roller_start_pose = gymapi.Transform()
        self.roller_start_pose.p.z = 0.06

    def load_robinion_asset(self):
        assets_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_robinion_path = "urdf/robinion_meshes/fixed_robinion_torso_center.urdf"
        asset_robinion_options = gymapi.AssetOptions()
        asset_robinion_options.fix_base_link = False
        asset_robinion_options.collapse_fixed_joints = False

        #Use vhacd to overcome the triangle meshes
        asset_robinion_options.vhacd_enabled = True
        asset_robinion_options.vhacd_params.resolution = 300000
        asset_robinion_options.vhacd_params.max_convex_hulls = 10
        asset_robinion_options.vhacd_params.max_num_vertices_per_ch = 64

        self.asset_robinion = self.gym.load_asset(self.sim, assets_root_dir, asset_robinion_path, asset_robinion_options)
        if self.debug_mode:
            balance_board_utils.print_asset_info(self.asset_robinion, "Robinion", self.gym)
            input(f"Here is the info of Robinion, press any key to continue")
        self.num_dof = self.gym.get_asset_dof_count(self.asset_robinion)
        
        self.robinion_start_pose = gymapi.Transform()
        self.robinion_start_pose.p.z = 0.72
        self.robinion_start_pose.r = gymapi.Quat(0.0, 0.0, -0.707107, 0.707107)

    def Generate_dict_for_indexing(self):
        #Rigid body dict 
        self.robinion_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset_robinion)
        for rigid_body_name in self.robinion_rigid_body_dict.keys():
            self.robinion_rigid_body_dict[rigid_body_name] += 1 #0 is roller
        #Dof dict
        self.dof_dict = self.gym.get_asset_dof_dict(self.asset_robinion)

        self.mode_joints_index_dict = {}
        for mode_joints in self.mode_joints_dict:
            controlled_dof_index = []
            for dof_name in self.mode_joints_dict[mode_joints]:
                controlled_dof_index.append(self.dof_dict[dof_name])
            controlled_dof_index = sorted(controlled_dof_index)
            self.mode_joints_index_dict[mode_joints + "_index"] = controlled_dof_index
            self.mode_joints_index_dict["fixed_" + mode_joints + "_index"] = list(set(range(self.num_dof)) - set(controlled_dof_index))

        self.controlled_joints_index = self.mode_joints_index_dict[self.mode_joint_name + "_dof_index"]
        self.fixed_joints_index = self.mode_joints_index_dict["fixed_" + self.mode_joint_name + "_dof_index"]

        #For parallel pitch mode
        if self.p_control_mode == ControlMode.parallel_pitch:
            self.l_knee_pitch_index = self.dof_dict["l_knee_pitch_joint"]
            self.r_knee_pitch_index = self.dof_dict["r_knee_pitch_joint"]
            self.l_ankle_pitch_index = self.dof_dict["l_ankle_pitch_joint"]
            self.r_ankle_pitch_index = self.dof_dict["r_ankle_pitch_joint"]
            self.l_knee_pitch_index_index = self.controlled_joints_index.index(self.dof_dict["l_knee_pitch_joint"])
            self.r_knee_pitch_index_index = self.controlled_joints_index.index(self.dof_dict["r_knee_pitch_joint"])

            knee_ankle_list = [self.l_knee_pitch_index, self.l_ankle_pitch_index, self.r_knee_pitch_index, self.r_ankle_pitch_index]
            self.list_transfer_hip_to_knee_ankle = []

            index = -1
            for dof_index in self.controlled_joints_index:
                if dof_index in knee_ankle_list:
                    self.list_transfer_hip_to_knee_ankle.append(index)
                else:
                    index += 1
                    self.list_transfer_hip_to_knee_ankle.append(index)

            self.parallel_pitch_without_knee_ankle = self.controlled_joints_index.copy()
            self.parallel_pitch_without_knee_ankle.remove(self.dof_dict["l_knee_pitch_joint"])
            self.parallel_pitch_without_knee_ankle.remove(self.dof_dict["r_knee_pitch_joint"])
            self.parallel_pitch_without_knee_ankle.remove(self.dof_dict["l_ankle_pitch_joint"])
            self.parallel_pitch_without_knee_ankle.remove(self.dof_dict["r_ankle_pitch_joint"])

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.load_roller_asset()
        self.load_robinion_asset()
        self.Generate_dict_for_indexing()
        
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.envs = []
        self.rollers = []
        self.robinions = []        

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            roller = self.gym.create_actor(env_ptr, self.asset_roller, self.roller_start_pose, "roller", i, 0, 0)
            robinion = self.gym.create_actor(env_ptr, self.asset_robinion, self.robinion_start_pose, "actor", i, 0, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, robinion)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][self.controlled_joints_index] = self.p_controlled_joints_stiffness
            dof_props["damping"][self.controlled_joints_index] = self.p_controlled_joints_damping
            dof_props["stiffness"][self.fixed_joints_index] = self.p_fixed_joints_stiffness
            dof_props["damping"][self.fixed_joints_index] = self.p_fixed_joints_damping
            
            self.gym.set_actor_dof_properties(env_ptr, robinion, dof_props)
            self.rollers.append(roller)
            self.robinions.append(robinion)
            self.envs.append(env_ptr)
        
        self.dof_props = dof_props
        self.upper_limit = torch.tensor(self.dof_props["upper"],
            device=self.device, dtype=torch.float32)
        self.lower_limit = torch.tensor(self.dof_props["lower"],
            device=self.device, dtype=torch.float32)
        self.dof_range = self.upper_limit - self.lower_limit

        if self.debug_mode == True:
            balance_board_utils.print_actor_info(self.gym, self.envs[0], self.robinions[0])

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.log_reward_tensor = compute_reward(
            self.reset_buf,
            self.root_tensor,
            self.rigid_body_state, 
            self.robinion_rigid_body_dict,
            self.roller_id, 
            self.robot_id,
            self.p_bongo_board_reward_weight,
            self.p_robot_reset_height_threshold,
            self.max_episode_length,
            self.progress_buf,
            self.obs_buf,
            self.log_reward_tensor
            )
        self.episode_sums["bongo_board"] += self.log_reward_tensor[0]
        self.episode_sums["bongo_vel"] += self.log_reward_tensor[2]
        self.episode_sums["upright"] += self.log_reward_tensor[1]
        self.episode_sums["l_foot_position"] += self.log_reward_tensor[3]

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.obs_buf[:] = compute_observations_sim(
            self.root_tensor,
            self.rigid_body_state,
            self.robinion_rigid_body_dict,
            self.dof_pos,
            self.dof_vel,
            self.robot_id,
            self.roller_id,
            self.z_basis_vec,
            self.controlled_joints_index,
            self.previous_actions,
            )

    def reset_envs(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        robot_reset_indexes = self.robot_id[env_ids]
        roller_reset_indexes = self.roller_id[env_ids]
        actor_reset_indexes = torch.cat([
            robot_reset_indexes, roller_reset_indexes
        ])
        actor_reset_indexes_int32 = actor_reset_indexes.to(torch.int32)
        
        self.root_tensor[actor_reset_indexes, :] = self.reset_root_tensor[actor_reset_indexes, :]
        roller_reset_height = (torch.rand((len(env_ids),), device=self.device) * \
             self.p_reset_range_height) + self.p_min_roller_height
        self.root_positions[roller_reset_indexes, 2] = roller_reset_height
        self.root_positions[robot_reset_indexes, 2] = roller_reset_height + self.p_robot_roller_reset_offset_height
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(actor_reset_indexes_int32),
            len(actor_reset_indexes))
        
        robot_reset_indexes_int32 = robot_reset_indexes.to(dtype=torch.int32)
        positions = torch_rand_float(-0.03, 0.03, (len(env_ids), self.num_dof), device=self.device)#0.03
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)#0.1
        self.dof_pos[env_ids] = tensor_clamp(self.reset_dof_pos[env_ids] + positions, self.lower_limit, self.upper_limit)
        self.dof_vel[env_ids] = velocities
        self.gym.set_dof_state_tensor_indexed(self.sim, 
        gymtorch.unwrap_tensor(self.dof_state_tensor),
           gymtorch.unwrap_tensor(robot_reset_indexes_int32),
           len(robot_reset_indexes_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = self.episode_sums[key][env_ids]
            self.episode_sums[key][env_ids] = 0.

    def pre_physics_step(self, actions):
        if self.p_control_mode == ControlMode.free:
            actions = self.dof_range[self.controlled_joints_index]/2*actions+((self.upper_limit+self.lower_limit)/2)[self.controlled_joints_index]

        elif self.p_control_mode == ControlMode.parallel_pitch: 
            rotating_degree_pitch_only = actions * self.dof_range[self.parallel_pitch_without_knee_ankle]/2 #multiply the dof range except for knees and ankles
            rotating_degree = rotating_degree_pitch_only[:, self.list_transfer_hip_to_knee_ankle] * self.parallel_pitch_weight_tensor  #Mapping and mutiply the weight for knee
            actions = rotating_degree + ((self.upper_limit + self.lower_limit)/2)[self.controlled_joints_index] #Add the middle of the joint

        elif self.p_control_mode == ControlMode.inverse_kinematics:
            pass

        self.pd_targets_tensor[:, self.controlled_joints_index] = actions
        self.pd_targets_tensor = saturate(self.pd_targets_tensor,
            self.lower_limit, self.upper_limit)
        self.gym.set_dof_position_target_tensor(self.sim, 
            gymtorch.unwrap_tensor(self.pd_targets_tensor))

        self.previous_actions = actions.clone() 

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        #Compute disturbance tolerance
        if len(reset_env_ids) > 0:
            self.reset_envs(reset_env_ids)
            if self.tolerance_disturbance_mode:
                self.total_force += sum(self.force_magnitude[reset_env_ids])
                print(self.total_force)
                self.force_magnitude[reset_env_ids] = 0
                self.fall_time += len(reset_env_ids)
                print(self.fall_time)
                print(self.total_force / self.fall_time)
                if self.fall_time >= 2048:
                    print(self.total_force / self.fall_time)
                    input()

        self.compute_observations()
        self.compute_reward()

        #Add 10N every 2 seconds
        if self.tolerance_disturbance_mode and self.tolerance_count % 120 == 0:
            print("Apply force:", self.force_magnitude)
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
            self.force_magnitude += 10
            self.forces[:, 1, :] = normalize(torch.randn((self.num_envs,3), device = self.device, dtype=torch.float32))
            self.forces[:, 1, :] *= self.force_magnitude.unsqueeze(-1)

        if self.tolerance_disturbance_mode:
            self.tolerance_count += 1

        #Compute average angular velocity of the board
        if self.board_pitch_mode:
            self.board_pitch_ang_vel_sum += self.rigid_body_state[0, self.robinion_rigid_body_dict["board"], 11]
            self.step_count += 1
            print(self.board_pitch_ang_vel_sum)
            self.average_board_pitch_ang_vel = self.board_pitch_ang_vel_sum / self.step_count
            print("step count", self.step_count, "Average ang vel:", self.average_board_pitch_ang_vel)
            if self.step_count == 3600:
                input()

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_reward(reset_buf, root_tensor, rigid_body_state, robinion_rigid_body_dict, roller_ids, robot_actor_ids, 
    p_bongo_board_reward_weight, p_robot_reset_height_threshold, max_episode_length, progress_buf, obs_buf, log_reward_tensor):
    # type: (Tensor, Tensor, Tensor, Dict[str, int], Tensor, Tensor, float, float, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    
    ones = torch.ones_like(reset_buf)
    robot_torso_height = root_tensor[robot_actor_ids, 2]
    bongo_board_angvels = rigid_body_state[:, robinion_rigid_body_dict["board"], 7:10]

    bongo_board_rpy = get_euler_xyz(rigid_body_state[:, robinion_rigid_body_dict["board"], 3:7])
    pitch = bongo_board_rpy[0]
    pitch = torch.where(pitch > np.pi, pitch % np.pi - np.pi, pitch)

    up_projection = obs_buf[:, -1]
    up_projection = torch.clip(up_projection, 0.0, 1.0)

    l_foot_position = rigid_body_state[:, robinion_rigid_body_dict["l_foot_link"], 0:3]
    Det_box_position = rigid_body_state[:, robinion_rigid_body_dict["DET_BOX_l_foot"], 0:3]
    l_foot_dist = torch.norm(l_foot_position - Det_box_position, p=2, dim=1)

    bongo_board_rew = 1 - (p_bongo_board_reward_weight * pitch ** 2)
    bongo_board_vel_rew = -torch.norm(bongo_board_angvels, dim=1)
    upright_rew = -(1 - up_projection)
    l_foot_dist_reward = 1 - 20 * l_foot_dist**2     
    rew_buf = bongo_board_rew + upright_rew + bongo_board_vel_rew + l_foot_dist
    
    log_reward_tensor[0] = bongo_board_rew
    log_reward_tensor[1] = upright_rew
    log_reward_tensor[2] = bongo_board_vel_rew
    log_reward_tensor[3] = l_foot_dist_reward

    yaw = bongo_board_rpy[2]
    yaw = torch.where(yaw > np.pi, yaw % np.pi - np.pi, yaw)
    yaw_degree = yaw * 180 / np.pi
    tolerate_degree = 5

    reset_buf[progress_buf >= max_episode_length] = 1
    reset_buf = torch.where(robot_torso_height < p_robot_reset_height_threshold, ones, reset_buf)
    reset_buf = torch.where(rigid_body_state[:, robinion_rigid_body_dict["L_detection_box"], 2] < 0.0 , ones, reset_buf)
    reset_buf = torch.where(rigid_body_state[:, robinion_rigid_body_dict["R_detection_box"], 2] < 0.0 , ones, reset_buf)
    reset_buf = torch.where(l_foot_dist > 0.08, ones, reset_buf)
    reset_buf = torch.where(yaw_degree < -90 - tolerate_degree, ones, reset_buf)
    reset_buf = torch.where(yaw_degree > -90 + tolerate_degree, ones, reset_buf)
    reset_buf[progress_buf == 0] = 0
    #Reset punishment
    rew_buf = torch.where(robot_torso_height < p_robot_reset_height_threshold,  ones * -1.0, rew_buf)
    rew_buf = torch.where(rigid_body_state[:, robinion_rigid_body_dict["L_detection_box"], 2] < 0.0 , ones * -1.0, rew_buf)
    rew_buf = torch.where(rigid_body_state[:, robinion_rigid_body_dict["R_detection_box"], 2] < 0.0 , ones * -1.0, rew_buf)
    rew_buf = torch.where(l_foot_dist > 0.08, ones * -1.0, rew_buf)
    rew_buf = torch.where(yaw_degree < -90 - tolerate_degree, ones * -1.0, rew_buf)
    rew_buf = torch.where(yaw_degree > -90 + tolerate_degree, ones * -1.0, rew_buf)

    return rew_buf, reset_buf, log_reward_tensor

@torch.jit.script
def compute_observations_sim(root_tensor, rigid_body_state, robinion_rigid_body_dict, dof_pos, dof_vel,
        robot_actor_ids, roller_ids, z_basis_vec, parallel_pitch_controlled_dof_index, previous_actions):
    # type: (Tensor, Tensor, Dict[str, int], Tensor, Tensor, Tensor, Tensor, Tensor, List[int], Tensor) ->  (Tensor)

    robot_angvel = root_tensor[robot_actor_ids, 10:]

    robot_quat = root_tensor[robot_actor_ids, 3:7]
    robot_rpy = get_euler_xyz(robot_quat)

    robot_position = root_tensor[robot_actor_ids, :3]
    bongo_board_position = rigid_body_state[:, robinion_rigid_body_dict["board"], 0:3]
    roller_position = root_tensor[roller_ids, 0:3]

    torso_position = rigid_body_state[:, robinion_rigid_body_dict["torso_link"], 0:3]
    l_foot_position = rigid_body_state[:, robinion_rigid_body_dict["l_foot_link"], 0:3]
    r_foot_position = rigid_body_state[:, robinion_rigid_body_dict["r_foot_link"], 0:3]
    l_foot_relative_to_torso = l_foot_position - torso_position
    r_foot_relative_to_torso = r_foot_position - torso_position

    bongo_board_rpy = get_euler_xyz(rigid_body_state[:, robinion_rigid_body_dict["board"], 3:7])
    bongo_board_angle = bongo_board_rpy[0]
    bongo_board_angle = torch.where(bongo_board_angle > np.pi, bongo_board_angle % np.pi - np.pi, bongo_board_angle)

    l_foot_position = rigid_body_state[:, robinion_rigid_body_dict["l_foot_link"], 0:3]
    Det_box_position = rigid_body_state[:, robinion_rigid_body_dict["DET_BOX_l_foot"], 0:3]
    l_foot_dist = torch.norm(l_foot_position - Det_box_position, p=2, dim=1)

    up_vec = get_basis_vector(robot_quat, z_basis_vec)
    up_projection = up_vec[:, 2]

    obs = torch.cat([
        dof_pos[:, parallel_pitch_controlled_dof_index], # Only observe joints that we control.
        dof_vel[:, parallel_pitch_controlled_dof_index], # Only observe joints that we control.
        previous_actions,
        robot_angvel,
        robot_rpy[0].unsqueeze(-1),
        robot_rpy[1].unsqueeze(-1),
        robot_rpy[2].unsqueeze(-1),
        robot_position,
        bongo_board_position,
        roller_position,
        l_foot_relative_to_torso,
        r_foot_relative_to_torso,
        bongo_board_angle.unsqueeze(-1),
        l_foot_dist.unsqueeze(-1),
        up_projection.unsqueeze(-1)
    ], dim=1)

    return obs