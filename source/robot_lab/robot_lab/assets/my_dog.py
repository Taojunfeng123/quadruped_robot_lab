# Copyright (c) 2024-2026 Ziqi Fan
# Copyright (c) 2025-2026 Junfeng Tao
# SPDX-License-Identifier: Apache-2.0

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLP, ActuatorNetMLPCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
#from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

MY_DOG_ACTUATOR_NET_PATH = f"source/robot_lab/robot_lab/assets/motor.pt"
MY_DOG_CALF_SPRING_COEFFS = {
    "FR_calf_joint": (-1.026204, 1.966818),
    "FL_calf_joint": (-1.270471, -2.476247),
    "RR_calf_joint": (-1.077613, 2.153850),
    "RL_calf_joint": (-1.113021, -2.225375),
}
##执行器网络，用于在仿真中模仿真实的电机行为。它使用一个多层感知器（MLP）来计算关节扭矩，基于关节位置误差和速度的历史数据。该网络还可以选择性地应用小腿关节的弹簧力。
class MyDogSpringActuatorNetMLP(ActuatorNetMLP):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        spring_coeffs = torch.zeros(self.num_joints, 2, device=self._device)
        if self.cfg.enable_calf_spring:
            joint_name_to_index = {joint_name: index for index, joint_name in enumerate(self.joint_names)}
            for joint_name, coeffs in MY_DOG_CALF_SPRING_COEFFS.items():
                joint_index = joint_name_to_index.get(joint_name)
                if joint_index is not None:
                    spring_coeffs[joint_index, 0] = coeffs[0]
                    spring_coeffs[joint_index, 1] = coeffs[1]
        self._spring_coeffs = spring_coeffs

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
        self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos
        self._joint_vel_history = self._joint_vel_history.roll(1, 1)
        self._joint_vel_history[:, 0] = joint_vel
        self._joint_vel[:] = joint_vel

        pos_input = torch.cat([self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        pos_input = pos_input.view(self._num_envs * self.num_joints, -1)
        vel_input = torch.cat([self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        vel_input = vel_input.view(self._num_envs * self.num_joints, -1)
        if self.cfg.input_order == "pos_vel":
            network_input = torch.cat([pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale], dim=1)
        elif self.cfg.input_order == "vel_pos":
            network_input = torch.cat([vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale], dim=1)
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        with torch.inference_mode():
            torques = self.network(network_input).view(self._num_envs, self.num_joints)
        self.computed_effort = torques * self.cfg.torque_scale

        spring_effort = self._spring_coeffs[:, 0].unsqueeze(0) * joint_pos + self._spring_coeffs[:, 1].unsqueeze(0)
        self.computed_effort = self.computed_effort + spring_effort

        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

##四足机器人的配置类
@configclass
class MyDogActuatorNetMLPCfg(ActuatorNetMLPCfg):
    class_type: type = MyDogSpringActuatorNetMLP
    enable_calf_spring: bool = True
MY_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"source/robot_lab/data/Robots/newdog7/urdf/dog_raydar.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.285),
        joint_pos={
            
            "FR_hip_joint": 0.0,
            "FL_hip_joint": -0.0,
            "RR_hip_joint": -0.0,
            "RL_hip_joint": 0.0,

            "FR_thigh_joint": 0.8,
            "FL_thigh_joint": -0.8,
            "RR_thigh_joint": 1.0,
            "RL_thigh_joint": -1.0,

            "FR_calf_joint": 0.8,
            "FL_calf_joint": -0.8,
            "RR_calf_joint": 0.9,
            "RL_calf_joint": -0.9,
        }, 
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ActuatorNetMLPCfg(
            joint_names_expr=[".*"],
            network_file=MY_DOG_ACTUATOR_NET_PATH,
            pos_scale=1.0, # 位置误差输入不变号、不缩放。
            vel_scale=1.0, # 关节速度输入不变号、不缩放。
            torque_scale=1.0, # 输出扭矩不缩放，保持与物理单位一致。
            input_order="pos_vel",
            input_idx=[0, 1, 2],
            effort_limit=16.8,
            saturation_effort=16.8,
            velocity_limit=32.5,
        ),
    },
)