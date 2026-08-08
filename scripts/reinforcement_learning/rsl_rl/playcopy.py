# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# Copyright (c) 2025-2026 Junfeng Tao
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections import deque
import math
import os
import sys
import numpy as np
import atexit
import torch
from isaaclab.app import AppLauncher
# 关闭调试可视化（修复变量未定义错误）
VIS_ENABLED = False
VIS_REF_ENABLE = False
cmd_filtered = torch.zeros(3)   # [vx, vy, wz]
cmd_target = torch.zeros(3)
decay_tau = 0.15   # 缓冲时间（关键参数）
# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

joint_names = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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

# import after SimulationApp is created to avoid early Omniverse/pxr imports
from rl_utils import camera_follow

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version
import isaaclab_tasks  # 这会自动注册所有任务
# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import time

import signal

import isaaclab.utils.math as math_utils

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
except Exception:
    try:
        import omni.isaac.debug_draw._debug_draw as omni_debug_draw
    except Exception:
        omni_debug_draw = None

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
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

# ===================== GLOBAL 统计变量（确保退出时能访问） =====================
torque_history = []
current_history = []
peak_current = 0.0
env_instance = None
# ==============================================================================

# ===================== 退出时强制输出统计报告 =====================
def print_stats_on_exit():
    global torque_history, current_history, peak_current
    if len(torque_history) == 0:
        return
    try:
        torque_array = np.array(torque_history)
        current_array = np.array(current_history)
        avg_current = np.mean(current_array)
        num_joints = torque_array.shape[1]
        joint_avg_torque = np.mean(np.abs(torque_array), axis=0)
        joint_peak_torque = np.max(np.abs(torque_array), axis=0)

        print("\n" + "="*70)
        print("           仿真已退出 —— 最终力矩 & 电流统计")
        print("="*70)
        print(f"总有效步数: {len(torque_array)}")
        print(f"平均电流: {avg_current:.2f} A")
        print(f"峰值电流: {peak_current:.2f} A")
        print("\n各关节力矩 (N·m):")
        print("-" * 50)
        for i in range(num_joints):
            name = joint_names[i] if i < len(joint_names) else f"Joint{i}"
            print(f"{name:<10} | 平均: {joint_avg_torque[i]:6.2f} | 峰值: {joint_peak_torque[i]:6.2f}")
        print("="*70 + "\n")
    except Exception as e:
        print("退出统计异常:", e)
        return

# 注册退出钩子（无论怎么关都会执行）
atexit.register(print_stats_on_exit)

# 处理关闭信号
def handle_exit_signal(signum, frame):
    global env_instance
    if env_instance is not None:
        env_instance.close()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)
# ====================================================================

def _convert_legacy_checkpoint(checkpoint_path: str) -> str:
    """Detect and convert legacy checkpoint to v5 format; return the path to load.

    Supports three checkpoint formats:
      A) Oldest  (rsl-rl < 4.0):  {"model_state_dict": {"actor.*", "critic.*", "log_std"}, ...}
      B) Split   (manual conv):   {"actor_state_dict": {"0.weight", ..., "log_std"}, ...}   (no mlp. prefix)
      C) Current (rsl-rl >= 4.0): {"actor_state_dict": {"mlp.*", "distribution.log_std_param"}, ...}

    The conversion is **non-destructive**: Format C is left untouched; formats A/B are
    converted and written to a ``*_v5_compat.pt`` sibling file.  The returned path is the
    one that can actually be loaded by the current rsl-rl runner.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── detect format ────────────────────────────────────────────────────────
    if "model_state_dict" in ckpt:
        fmt = "A"
    elif "actor_state_dict" in ckpt:
        first_key = next(iter(ckpt["actor_state_dict"].keys()), "")
        if first_key.startswith("mlp.") or first_key.startswith("rnn.") or first_key.startswith("distribution."):
            fmt = "C"
        else:
            fmt = "B"
    else:
        fmt = "C"  # unknown – trust the caller

    if fmt == "C":
        return checkpoint_path  # nothing to do

    # ── convert ──────────────────────────────────────────────────────────────
    print(f"[INFO] Detected legacy checkpoint format ({fmt}), converting to v5 ...")

    def _is_rnn_key(k: str) -> bool:
        return k.startswith("rnn.") or k in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers")

    def _to_mlpmodel(state: dict) -> dict:
        """Rewrite flat/old keys into MLPModel / RNNModel nested keys."""
        new_state: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k == "log_std":
                new_state["distribution.log_std_param"] = v
            elif _is_rnn_key(k):
                new_state[k] = v  # rnn.* keys stay as-is (already prefixed)
            else:
                new_state[f"mlp.{k}"] = v
        return new_state

    if fmt == "A":
        model_dict = ckpt.pop("model_state_dict")
        actor_raw: dict[str, torch.Tensor] = {}
        critic_raw: dict[str, torch.Tensor] = {}

        for k, v in model_dict.items():
            if k.startswith("actor."):
                actor_raw[k.removeprefix("actor.")] = v
            elif k.startswith("critic."):
                critic_raw[k.removeprefix("critic.")] = v
            elif k == "log_std":
                actor_raw[k] = v
            # ignore other keys (e.g. standalone rnn.* from old ActorCriticRecurrent)

        ckpt["actor_state_dict"] = _to_mlpmodel(actor_raw)
        ckpt["critic_state_dict"] = _to_mlpmodel(critic_raw)

    elif fmt == "B":
        ckpt["actor_state_dict"] = _to_mlpmodel(ckpt["actor_state_dict"])
        ckpt["critic_state_dict"] = _to_mlpmodel(ckpt["critic_state_dict"])

    # ── write compat file ────────────────────────────────────────────────────
    out_path = checkpoint_path.replace(".pt", "_v5_compat.pt")
    torch.save(ckpt, out_path)
    print(f"[INFO] Converted checkpoint saved to: {out_path}")
    return out_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    global torque_history, current_history, peak_current, env_instance
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # handle deprecated configurations (convert old policy format to new actor/critic format)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
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
    env_cfg.curriculum.command_levels = None

    keyboard_command_state = None
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

        def _keyboard_obs_term(env):
            nonlocal keyboard_command_state
            keyboard_command_state = torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device)
            return keyboard_command_state

        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=_keyboard_obs_term,
        )
        '''def _keyboard_obs_term(env):
            nonlocal keyboard_command_state
            global cmd_filtered, cmd_target

            raw_cmd = torch.tensor(controller.advance(), dtype=torch.float32)

            # 更新目标命令
            cmd_target = raw_cmd

        # ===== 核心：一阶低通滤波（实现缓冲）=====
            dt = env.step_dt
            alpha = dt / (decay_tau + dt)
            alphax=dt/(0.1+dt)
            #cmd_filtered = (1 - alpha) * cmd_filtered + alpha * cmd_target
            # vx 做衰减更快的滤波
            cmd_filtered[0] = (1-alphax)*cmd_filtered[0]+alphax*cmd_target[0]

            # 只对 vy 做低通滤波（横向缓慢变化）
            cmd_filtered[1] = (1 - alpha) * cmd_filtered[1] + alpha * cmd_target[1]

            # wz 直接用目标值（不滤波）
            cmd_filtered[2] = cmd_target[2]
            keyboard_command_state = cmd_filtered.unsqueeze(0).to(env.device)
            return keyboard_command_state'''
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=_keyboard_obs_term,
        )
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env_instance = env

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "video_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # convert legacy checkpoint format (rsl-rl < 5.0 -> >= 5.0) if needed
    resume_path = _convert_legacy_checkpoint(resume_path)
    # load previously trained model
    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # Use runner-native exporters for rsl-rl >= 4.0.0
        ppo_runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        ppo_runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        # Fallback for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = ppo_runner.alg.policy
        else:
            policy_nn = ppo_runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        else:
            normalizer = None

        export_policy_as_onnx(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        export_policy_as_jit(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )

    dt = env.unwrapped.step_dt
    # reset environment
    obs, _ = env.reset()
    
    timestep = 0
    # 清空统计
    torque_history.clear()
    current_history.clear()
    peak_current = 0.0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        
        # 获取真实力矩
        applied_torque = env.unwrapped.scene["robot"].data.applied_torque
        current = applied_torque.abs().sum().cpu().item() / 1.22 * 1.1
        
        # 保存统计（一定会存）
        torque_history.append(applied_torque[0].cpu().numpy().copy())
        current_history.append(current)
        if current > peak_current:
            peak_current = current
        # 获取 base 速度
        root_state = env.unwrapped.scene["robot"].data.root_state_w

        vx = root_state[0, 7].item()
        vy = root_state[0, 8].item()
        # 实时打印
        print("="*50)
        print(f"线速度 -> vx: {vx:.3f} m/s, vy: {vy:.3f} m/s")
        print(f"实时力矩:\n{applied_torque[0].cpu().numpy()}")
        print(f"实时电流: {current:.2f} A")

        # 视频逻辑
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # 键盘相机跟随
        if args_cli.keyboard:
            camera_follow(env)

        # 实时帧率
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()