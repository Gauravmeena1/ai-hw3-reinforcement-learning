# train_mountaincar.py
# HW3 - DQN / PPO on MountainCar-v0 and MountainCarContinuous-v0
# Usage:
#   python train_mountaincar.py --config dqn_no_shaping   --timesteps 200000
#   python train_mountaincar.py --config dqn_with_shaping --timesteps 200000
#   python train_mountaincar.py --config ppo_continuous   --timesteps 200000
#
# Refs:
#   Ng et al., "Policy invariance under reward transformations," ICML 1999
#     (theoretical basis for potential-based reward shaping)
#   stable-baselines3 (https://stable-baselines3.readthedocs.io)

import os
import argparse
import torch
import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from config import (
    MOUNTAINCAR_CONFIGS,
    MOUNTAINCAR_DQN_FIXED,
    MOUNTAINCAR_PPO_FIXED,
    MOUNTAINCAR_TOTAL_TIMESTEPS,
)


# Reward Shaping Wrapper

class PotentialBasedShaping(gym.Wrapper):
    """Potential-based reward shaping for MountainCar-v0.

    By Ng et al. (1999), any shaping of the form
        F(s, s') = γ·Φ(s') − Φ(s)
    leaves the optimal policy unchanged while making the reward signal denser.

    Potential function:
        Φ(s) = w_pos · position + w_vel · |velocity|

    Intuition:
      - Reward the agent for moving right (increasing position).
      - Reward the agent for building speed (the car must rock back and forth).
      - The velocity term is the critical signal: a stopped car at x = −0.5 gets
        no bonus, so the agent learns to keep oscillating.
    """

    def __init__(self, env, w_pos: float = 1.0, w_vel: float = 10.0, gamma: float = 0.99):
        super().__init__(env)
        self.w_pos   = w_pos
        self.w_vel   = w_vel
        self.gamma   = gamma
        self._phi_prev = 0.0

    def _phi(self, obs) -> float:
        """Compute potential Φ(s) from observation."""
        position, velocity = float(obs[0]), float(obs[1])
        return self.w_pos * position + self.w_vel * abs(velocity)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi_prev = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_curr = self._phi(obs)
        # F = γ·Φ(s') − Φ(s)  (potential difference)
        shaping  = self.gamma * phi_curr - self._phi_prev
        self._phi_prev = phi_curr
        return obs, reward + shaping, terminated, truncated, info


# Environment factories

def _make_discrete_env(reward_shaping: bool, seed: int):
    """Factory for Monitor-wrapped MountainCar-v0 (discrete, 3 actions)."""
    def _init():
        env = gym.make("MountainCar-v0")
        if reward_shaping:
            env = PotentialBasedShaping(env)
        env = Monitor(env)   # records episode returns/lengths to CSV
        return env
    return _init


def _make_continuous_env(seed: int):
    """Factory for Monitor-wrapped MountainCarContinuous-v0 (continuous action)."""
    def _init():
        env = gym.make("MountainCarContinuous-v0")
        env = Monitor(env)
        return env
    return _init


# DQN trainer

def train_dqn(config_name: str,
              reward_shaping: bool,
              learning_rate: float,
              total_timesteps: int,
              seed: int) -> DQN:
    """Train DQN on discrete MountainCar-v0."""

    print(f"\n{'='*60}")
    print(f" Training: MountainCar-v0 | DQN | config={config_name}")
    print(f"  reward_shaping={reward_shaping}  lr={learning_rate}")
    print(f"  timesteps={total_timesteps:,}  seed={seed}")
    print(f"{'='*60}")

    log_dir   = f"logs/mountaincar/{config_name}"
    model_dir = f"models/mountaincar/{config_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Training env uses shaping; eval env NEVER uses shaping so rewards are
    # comparable (raw environment reward) across all three configurations.
    train_env = DummyVecEnv([_make_discrete_env(reward_shaping, seed)])
    eval_env  = DummyVecEnv([_make_discrete_env(False, seed + 100)])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = model_dir,
        log_path             = log_dir,
        eval_freq            = 10_000,
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 1,
    )

    fixed  = MOUNTAINCAR_DQN_FIXED
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DQN(
        policy              = "MlpPolicy",
        env                 = train_env,
        learning_rate       = learning_rate,
        buffer_size         = fixed["buffer_size"],
        learning_starts     = fixed["learning_starts"],
        batch_size          = fixed["batch_size"],
        gamma               = fixed["gamma"],
        train_freq          = fixed["train_freq"],
        gradient_steps      = fixed["gradient_steps"],
        target_update_interval = fixed["target_update_interval"],
        exploration_fraction   = fixed["exploration_fraction"],
        exploration_initial_eps = fixed["exploration_initial_eps"],
        exploration_final_eps   = fixed["exploration_final_eps"],
        verbose             = 1,
        seed                = seed,
        device              = device,
        tensorboard_log     = f"tensorboard/mountaincar/{config_name}",
    )

    model.learn(
        total_timesteps = total_timesteps,
        callback        = eval_cb,
        progress_bar    = True,
    )
    model.save(f"{model_dir}/final_model")
    print(f"\n  Saved → {model_dir}/final_model.zip")

    train_env.close()
    eval_env.close()
    return model


# PPO trainer

def train_ppo(config_name: str,
              learning_rate: float,
              total_timesteps: int,
              seed: int) -> PPO:
    """Train PPO on continuous MountainCarContinuous-v0."""

    print(f"\n{'='*60}")
    print(f" Training: MountainCarContinuous-v0 | PPO | config={config_name}")
    print(f"  lr={learning_rate}  timesteps={total_timesteps:,}  seed={seed}")
    print(f"{'='*60}")

    log_dir   = f"logs/mountaincar/{config_name}"
    model_dir = f"models/mountaincar/{config_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_env = DummyVecEnv([_make_continuous_env(seed)])
    eval_env  = DummyVecEnv([_make_continuous_env(seed + 100)])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = model_dir,
        log_path             = log_dir,
        eval_freq            = 10_000,
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 1,
    )

    fixed  = MOUNTAINCAR_PPO_FIXED
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = learning_rate,
        n_steps         = fixed["n_steps"],
        batch_size      = fixed["batch_size"],
        n_epochs        = fixed["n_epochs"],
        gamma           = fixed["gamma"],
        gae_lambda      = fixed["gae_lambda"],
        clip_range      = fixed["clip_range"],
        ent_coef        = fixed["ent_coef"],
        vf_coef         = fixed["vf_coef"],
        max_grad_norm   = fixed["max_grad_norm"],
        verbose         = 1,
        seed            = seed,
        device          = device,
        tensorboard_log = f"tensorboard/mountaincar/{config_name}",
    )

    model.learn(
        total_timesteps = total_timesteps,
        callback        = eval_cb,
        progress_bar    = True,
    )
    model.save(f"{model_dir}/final_model")
    print(f"\n  Saved → {model_dir}/final_model.zip")

    train_env.close()
    eval_env.close()
    return model


# CLI entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN/PPO on MountainCar variants (HW3)"
    )
    parser.add_argument(
        "--config", type=str, default="dqn_with_shaping",
        choices=list(MOUNTAINCAR_CONFIGS.keys()),
    )
    parser.add_argument("--timesteps", type=int, default=MOUNTAINCAR_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg    = MOUNTAINCAR_CONFIGS[args.config].copy()
    method = cfg.pop("method")
    cfg.pop("env_id", None)                   # env is fixed per method

    if method == "dqn":
        train_dqn(
            config_name    = args.config,
            reward_shaping = cfg["reward_shaping"],
            learning_rate  = cfg["learning_rate"],
            total_timesteps = args.timesteps,
            seed           = args.seed,
        )
    else:
        train_ppo(
            config_name    = args.config,
            learning_rate  = cfg["learning_rate"],
            total_timesteps = args.timesteps,
            seed           = args.seed,
        )
