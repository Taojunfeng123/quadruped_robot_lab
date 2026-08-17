# Copyright (c) 2024-2025 Ziqi Fan
# Copyright (c) 2025-2026 Junfeng Tao
# SPDX-License-Identifier: Apache-2.0


from isaaclab.utils import configclass


import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
# from isaaclab.sensors.ray_caster import GridPatternCfg
##
# Pre-defined configs
##
from robot_lab.assets.my_dog import MY_DOG_CFG 
@configclass
class RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_calf_link"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]  # 含义：策略观测与动作控制的关节白名单；目标趋势：保持关节顺序稳定，减少训练输入输出漂移。
    # fmt: on  关闭格式化，保持关节名称列表的清晰结构

    link_names = [
    "base_link",

    "FL_hip_link", "FR_hip_link", "RL_hip_link", "RR_hip_link",
    "FL_thigh_link", "FR_thigh_link", "RL_thigh_link", "RR_thigh_link",
    "FL_calf_link", "FR_calf_link", "RL_calf_link", "RR_calf_link",
    ]

    hipx_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    ]

    hipy_joint_names = [
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    ]

    knee_joint_names = [
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs=3000
        # ------------------------------Sence------------------------------
        self.scene.robot = MY_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.4
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.8
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        
        self.observations.policy.height_scan = None # type: ignore
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        #self.observations.policy.base_lin_vel = None  # 含义：策略端线速度观测项；目标趋势：\
        self.observations.policy.base_lin_vel=None  # 线速度缩放
        #self.observations.policy.base_lin_vel.clip = (-100.0, 100.0)
        #self.observations.policy.height_scan = None 
        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}  # 含义：动作裁剪范围；目标趋势：保持宽裁剪避免过早饱和。
        self.actions.joint_pos.joint_names = self.joint_names  # 含义：动作生效关节；目标趋势：严格限制到目标关节集。

        # ------------------------------Events------------------------------
        self.scene.robot.init_state.pos = (0, 0, 0.30)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.1, 0.1),     # 极小随机：仅防止 overfitting
                "y": (-1, 0.1),     # 极小随机
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),  # 几乎水平
                "pitch": (-0.05, 0.05), # 几乎水平
                "yaw": (1.57, 1.57),     # 朝向 +Y（缝隙沿 Y 交替，前进方向为 Y）
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }


        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = self.link_names # [self.base_link_name]
        # self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name # [self.base_link_name]
        # self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.periodic_stop_command.params["cycle_period"]=3
        self.events.periodic_stop_command.params["stop_duration"]=1
        # ------------------------------Rewards------------------------------
        self.rewards.action_rate_l2.weight = -0.1 #-0.02
        # self.rewards.smoothness_2.weight = -0.0075

        self.rewards.base_height_l2.weight = -50.0
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.feet_air_time_lin_xy.weight = 5.0 # 5.0
        self.rewards.feet_air_time_lin_xy.params["threshold"] = 0.5
        self.rewards.feet_air_time_lin_xy.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_ang_z.weight = 5.0 # 5.0
        self.rewards.feet_air_time_ang_z.params["threshold"] = 0.5
        self.rewards.feet_air_time_ang_z.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.0 # -8.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.weight = -2.0 # -10.0
        self.rewards.foot_impact_velocity.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.stand_still.weight = -20 # -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.feet_height_body.weight = -0.0 # -2.5
        self.rewards.feet_height_body.params["target_height"] = -0.35
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.0 # -0.2
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.contact_forces.weight = -1e-1 # -2e-2
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.lin_vel_z_l2.weight = -20.0 #-2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25 # -0.05
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.joint_torques_l2.weight = -2.5e-4
        self.rewards.joint_acc_l2.weight = -1e-8
        self.rewards.joint_power.weight = -8e-4
        self.rewards.flat_orientation_l2.weight = -20.0
        # add the following rewards to improve the gait
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_calf_link", "RR_calf_link"],
            ["FR_calf_link", "RL_calf_link"]
        ]
        self.rewards.phase_foot_trajectory_exp.weight = 2.0
        self.rewards.phase_foot_trajectory_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ]
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.feet_contact_without_cmd.weight = 1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]

        # added rewards
        self.rewards.hipx_joint_pos_penalty.weight = -2
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -1.5
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "RoughEnvCfg":
            self.disable_zero_weight_rewards()
        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip_link"]  # 含义：非法接触终止检测体；目标趋势：重点约束躯干与髋部碰撞。
        #self.terminations.bad_orientation_2 = None
        self.terminations.illegal_contact=None
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels.params["range_multiplier"] = (0.5, 1.0)
        self.curriculum.command_levels.params["stop_cycle_period"] = 3   # 与 periodic_stop_command 保持一致
        self.curriculum.command_levels.params["stop_duration"] = 1       # 与 periodic_stop_command 保持一致
        self.curriculum.terrain_levels.params["stop_cycle_period"] = 3   # 与 periodic_stop_command 保持一致
        self.curriculum.terrain_levels.params["stop_duration"] = 1       # 与 periodic_stop_command 保持一致'''
        # ------------------------------Commands------------------------------
        #self.commands.base_velocity=None
        self.commands.base_velocity.ranges.lin_vel_x = (-2, 2)
        self.commands.base_velocity.ranges.lin_vel_y = (-1, 1)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.6, 1.6)