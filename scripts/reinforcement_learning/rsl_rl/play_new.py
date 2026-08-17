# Copyright (c) 2024-2026 Ziqi Fan
# Copyright (c) 2025-2026 Junfeng Tao
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _convert_rsl_rl_checkpoint(checkpoint_path: str, noise_std_type: str = "scalar") -> str:
    """Make checkpoints from other rsl-rl versions loadable by the installed rsl-rl 2.3.x.

    The installed rsl-rl (2.3.3) expects checkpoints of the form
    ``{"model_state_dict": {"std"|"log_std", "actor.*", "critic.*"}, "optimizer_state_dict": ...,
    "iter": ..., "infos": ...}`` where the noise parameter is named after the policy's
    ``noise_std_type`` ("scalar" -> ``std``, "log" -> ``log_std``). Newer rsl-rl (>= 3.0)
    stores the actor/critic states separately with ``mlp.``-prefixed keys and
    ``distribution.log_std_param`` (log-scale), which makes ``runner.load()`` fail with
    ``KeyError: 'model_state_dict'``.

    This detects the checkpoint format and converts it to the 2.3.x format for the given
    ``noise_std_type``. The conversion is non-destructive: a ``*_v2_compat.pt`` sibling file
    is written and returned. Checkpoints already in the matching 2.3.x format are returned
    unchanged.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if noise_std_type not in ("scalar", "log"):
        raise ValueError(f"Unsupported noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'.")
    target_noise_key = "std" if noise_std_type == "scalar" else "log_std"

    def _fix_noise_value(key: str, value) -> torch.Tensor:
        """Convert the stored noise parameter to the target scale.

        ``std`` (2.3.x scalar) and ``log_std``/``log_std_param`` (2.3.x log / >= 3.0) differ:
        the former stores the standard deviation linearly, the latter in log-space.
        """
        is_log_src = key in ("log_std", "distribution.log_std_param", "log_std_param")
        if noise_std_type == "log":
            return value if is_log_src else torch.log(value.clamp(min=1e-6))
        return value if not is_log_src else torch.exp(value)

    def _fix_policy_keys(state: dict) -> dict:
        """Map newer policy key naming to the 2.3.x ActorCritic naming."""
        out = {}
        for k, v in state.items():
            if k in ("std", "log_std", "distribution.log_std_param", "log_std_param"):
                out[target_noise_key] = _fix_noise_value(k, v)
            elif k.startswith("actor.mlp."):
                out["actor." + k.removeprefix("actor.mlp.")] = v
            elif k.startswith("critic.mlp."):
                out["critic." + k.removeprefix("critic.mlp.")] = v
            elif k.startswith("policy."):
                out[k.removeprefix("policy.")] = v
            else:
                out[k] = v
        return out

    if "model_state_dict" in ckpt:
        msd = _fix_policy_keys(ckpt["model_state_dict"])
        if msd.keys() == ckpt["model_state_dict"].keys():
            # nothing changed -> already in the matching 2.3.x format
            return checkpoint_path
        converted = dict(ckpt)
        converted["model_state_dict"] = msd
    elif "actor_state_dict" in ckpt and "critic_state_dict" in ckpt:
        # rsl-rl >= 4.0 (or pre-converted split) format
        msd = {}
        for k, v in ckpt["actor_state_dict"].items():
            if k in ("std", "log_std", "distribution.log_std_param", "log_std_param"):
                msd[target_noise_key] = _fix_noise_value(k, v)
            elif k.startswith("mlp."):
                msd["actor." + k.removeprefix("mlp.")] = v
            else:
                # rnn.* and other policy-level keys stay at policy level
                msd[k] = v
        for k, v in ckpt["critic_state_dict"].items():
            if k.startswith("mlp."):
                msd["critic." + k.removeprefix("mlp.")] = v
            else:
                msd[k] = v
        converted = {k: v for k, v in ckpt.items() if k not in ("actor_state_dict", "critic_state_dict")}
        converted["model_state_dict"] = msd
    else:
        # unknown format - trust the caller
        return checkpoint_path

    base = checkpoint_path[:-3] if checkpoint_path.endswith(".pt") else checkpoint_path
    out_path = base + "_v2_compat.pt"
    torch.save(converted, out_path)
    print(f"[INFO] Detected non-native rsl-rl checkpoint format, converted for rsl-rl 2.3.x: {out_path}")
    return out_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    if hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.command_levels_lin_vel = None
        env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # convert checkpoint to the installed rsl-rl version's format if necessary
    resume_path = _convert_rsl_rl_checkpoint(resume_path, noise_std_type=agent_cfg.policy.noise_std_type)
    runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module for export
    policy_nn = runner.alg.policy

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
