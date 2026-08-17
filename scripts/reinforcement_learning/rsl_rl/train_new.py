# Copyright (c) 2024-2026 Ziqi Fan
# Copyright (c) 2025-2026 Junfeng Tao
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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

import gymnasium as gym
import os
import time
import torch
from datetime import datetime

import omni
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # convert checkpoint to the installed rsl-rl version's format if necessary
        resume_path = _convert_rsl_rl_checkpoint(resume_path, noise_std_type=agent_cfg.policy.noise_std_type)
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
