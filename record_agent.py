# record_agent.py
# HW3 - Record MP4 videos of trained agents.
# Usage:
#   python record_agent.py --env spaceinvaders --config lr_mid
#   python record_agent.py --env mountaincar   --config dqn_with_shaping
#   python record_agent.py --env mountaincar   --config ppo_continuous

import os
import argparse

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    VecFrameStack, VecVideoRecorder, DummyVecEnv
)

from train_mountaincar import _make_discrete_env, _make_continuous_env
from config import SPACEINVADERS_FIXED


def record_spaceinvaders(config_name: str = "lr_mid", n_steps: int = 3_000):
    """Record the SpaceInvaders agent for n_steps frames and save as MP4."""
    model_path = f"models/spaceinvaders/{config_name}/best_model"
    if not os.path.exists(model_path + ".zip"):
        model_path = f"models/spaceinvaders/{config_name}/final_model"

    n_stack  = SPACEINVADERS_FIXED["n_stack"]
    video_dir = f"videos/spaceinvaders_{config_name}"
    os.makedirs(video_dir, exist_ok=True)

    env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecVideoRecorder(
        env, video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=n_steps,
        name_prefix=f"agent_{config_name}",
    )

    print(f"Loading model: {model_path}.zip")
    model = DQN.load(model_path, env=env)

    obs = env.reset()
    total_reward = 0
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    env.close()
    print(f"Done. Total reward: {total_reward:.1f}")
    print(f"Video saved to: {video_dir}/")


def record_mountaincar(config_name: str = "dqn_with_shaping", n_steps: int = 500):
    """Record the MountainCar agent and save as MP4."""
    model_path = f"models/mountaincar/{config_name}/best_model"
    if not os.path.exists(model_path + ".zip"):
        model_path = f"models/mountaincar/{config_name}/final_model"

    is_ppo    = "ppo" in config_name
    video_dir = f"videos/mountaincar_{config_name}"
    os.makedirs(video_dir, exist_ok=True)

    if is_ppo:
        base_env = DummyVecEnv([_make_continuous_env(seed=0)])
    else:
        base_env = DummyVecEnv([_make_discrete_env(reward_shaping=False, seed=0)])

    env = VecVideoRecorder(
        base_env, video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=n_steps,
        name_prefix=f"agent_{config_name}",
    )

    print(f"Loading model: {model_path}.zip")
    model_cls = PPO if is_ppo else DQN
    model = model_cls.load(model_path, env=env)

    obs = env.reset()
    total_reward = 0
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    env.close()
    print(f"Done. Total reward: {total_reward:.1f}")
    print(f"Video saved to: {video_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record trained RL agents (HW3)")
    parser.add_argument("--env", type=str, default="spaceinvaders",
                        choices=["spaceinvaders", "mountaincar"])
    parser.add_argument("--config", type=str, default="lr_mid")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()

    if args.env == "spaceinvaders":
        record_spaceinvaders(args.config, args.steps)
    else:
        record_mountaincar(args.config, args.steps)
