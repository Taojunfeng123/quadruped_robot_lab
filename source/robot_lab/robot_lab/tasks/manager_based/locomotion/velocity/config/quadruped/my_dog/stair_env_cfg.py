# Copyright (c) 2026 taojunfeng123
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025 Ziqi Fan
# Copyright (c) 2025-2026 Junfeng Tao
# SPDX-License-Identifier: Apache-2.0
from isaaclab.utils import configclass
from isaaclab.terrains import HfBridgeGapTerrainCfg
from .rough_env_cfg import RoughEnvCfg

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import RoughEnvCfg
# from isaaclab.sensors.ray_caster import GridPatternCfg
##
# Pre-defined configs
##
@configclass
class StairEnvCfg(RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
                # 断裂路面地形：核心场景
        self.scene.terrain.terrain_generator.sub_terrains["bridge_gap"] = HfBridgeGapTerrainCfg(
            proportion=0,
            size=(3.0, 10.0),    # X 窄 3m → 去掉两侧多余的平地；Y 保持 10m
            # 课程：difficulty 0→1，缝隙 5cm→15cm（平台宽度固定 40cm）
            gap_width_range=(0.05, 0.15),
            platform_strip_width_range=(0.40, 0.40),
            border_platform_width=1.0,
            holes_depth=-10.0,
        )

        ##self.rewards.feet_slide.weight = -0.2
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.lin_vel_z_l2.weight = -10.0 #-2.0
        self.rewards.flat_orientation_l2.weight = -15.0

        self.rewards.feet_air_time_lin_xy.weight = 5.0 # 5.0
        self.rewards.feet_air_time_lin_xy.params["threshold"] = 0.5
        self.rewards.feet_air_time_lin_xy.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_ang_z.weight = 5.0 # 5.0
        self.rewards.feet_air_time_ang_z.params["threshold"] = 0.5
        self.rewards.feet_air_time_ang_z.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.0 # -8.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.4
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.4
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.8
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "StairEnvCfg":
            self.disable_zero_weight_rewards()
        self.curriculum.command_levels.params["range_multiplier"] = (0.8, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"]=(3.2,3.2)
        self.terminations.illegal_contact=None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1, 1)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.6, 1.6)