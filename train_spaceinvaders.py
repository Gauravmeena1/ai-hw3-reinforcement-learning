# train_spaceinvaders.py
# HW3 - DQN on ALE/SpaceInvaders-v5
# Usage:  python train_spaceinvaders.py --config lr_mid --timesteps 1000000
# Ref:    stable-baselines3 (https://stable-baselines3.readthedocs.io)
#         Mnih et al., "Human-level control through deep RL," Nature 2015

import os
import argparse
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)

from config import SPACEINVADERS_CONFIGS, SPACEINVADERS_FIXED, SPACEINVADERS_TOTAL_TIMESTEPS


def make_envs(n_envs: int = 1, seed: int = 42, eval_seed: int = 142):
    """Create training and evaluation Atari environments.

    make_atari_env applies the standard preprocessing pipeline:
      - NoopResetEnv       -> random no-op starts for diversity
      - MaxAndSkipEnv      -> 4-frame action repeat + max-pooling
      - EpisodicLifeEnv    -> treat each life as an episode
      - FireResetEnv       -> press FIRE to start (SpaceInvaders needs this)
      - WarpFrame           -> resize to 84×84 grayscale
      - ClipRewardEnv      -> clip rewards to {-1, 0, +1}
    VecFrameStack then stacks the last 4 frames into one observation.
    """
    n_stack = SPACEINVADERS_FIXED["n_stack"]

    train_env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=n_envs, seed=seed)
    train_env = VecFrameStack(train_env, n_stack=n_stack)

    eval_env  = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1, seed=eval_seed)
    eval_env  = VecFrameStack(eval_env, n_stack=n_stack)

    return train_env, eval_env


def build_model(config_name: str, env, config: dict, seed: int) -> DQN:
    """Instantiate a DQN with CnnPolicy using the given hyperparameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {device}" + (
        f" — {torch.cuda.get_device_name(0)}" if device == "cuda" else ""
    ))

    fixed = SPACEINVADERS_FIXED
    return DQN(
        policy              = "CnnPolicy",
        env                 = env,
        # Variable hyperparameters (ablation targets)
        learning_rate       = config["learning_rate"],
        buffer_size         = config["buffer_size"],
        exploration_final_eps = config["exploration_final_eps"],
        # Fixed hyperparameters
        learning_starts     = fixed["learning_starts"],
        batch_size          = fixed["batch_size"],
        gamma               = fixed["gamma"],
        train_freq          = fixed["train_freq"],
        gradient_steps      = fixed["gradient_steps"],
        target_update_interval = fixed["target_update_interval"],
        exploration_fraction   = fixed["exploration_fraction"],
        exploration_initial_eps = fixed["exploration_initial_eps"],
        verbose             = 1,
        seed                = seed,
        device              = device,
        tensorboard_log     = f"tensorboard/spaceinvaders/{config_name}",
    )


def train(config_name: str,
          total_timesteps: int = SPACEINVADERS_TOTAL_TIMESTEPS,
          seed: int = 42) -> DQN:
    """Full training loop for one SpaceInvaders configuration."""

    config = SPACEINVADERS_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f" Training: SpaceInvaders | config={config_name}")
    print(f"  lr={config['learning_rate']}  "
          f"buffer={config['buffer_size']}  "
          f"eps_end={config['exploration_final_eps']}")
    print(f"  timesteps={total_timesteps:,}  seed={seed}")
    print(f"{'='*60}")

    # Directories
    log_dir   = f"logs/spaceinvaders/{config_name}"
    model_dir = f"models/spaceinvaders/{config_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Environments
    train_env, eval_env = make_envs(seed=seed)

    # Callbacks
    # EvalCallback: runs n_eval_episodes episodes on eval_env every eval_freq
    # steps, saves the best model, and writes evaluations.npz for plotting.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = model_dir,
        log_path             = log_dir,
        eval_freq            = 50_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )
    # CheckpointCallback: periodic snapshots so training can be resumed.
    ckpt_cb = CheckpointCallback(
        save_freq  = 200_000,
        save_path  = model_dir,
        name_prefix= "ckpt",
        verbose    = 1,
    )
    callbacks = CallbackList([eval_cb, ckpt_cb])

    # Model
    model = build_model(config_name, train_env, config, seed)

    # Train
    model.learn(
        total_timesteps = total_timesteps,
        callback        = callbacks,
        progress_bar    = True,
        reset_num_timesteps = True,
    )

    # Save final model alongside best model
    model.save(f"{model_dir}/final_model")
    print(f"\n  Saved → {model_dir}/final_model.zip")

    train_env.close()
    eval_env.close()
    return model


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN on SpaceInvaders (HW3)"
    )
    parser.add_argument(
        "--config", type=str, default="lr_mid",
        choices=list(SPACEINVADERS_CONFIGS.keys()),
        help="Which hyperparameter configuration to run",
    )
    parser.add_argument(
        "--timesteps", type=int, default=SPACEINVADERS_TOTAL_TIMESTEPS,
        help="Total environment steps",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.config, total_timesteps=args.timesteps, seed=args.seed)
