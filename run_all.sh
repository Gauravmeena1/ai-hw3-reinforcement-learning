#!/bin/bash
# run_all.sh - Execute all HW3 experiments sequentially.
#
# Estimated runtimes on an NVIDIA GPU:
#   SpaceInvaders (7 configs × 1M steps) about 7–10 hours total
#   MountainCar   (3 configs × 200K steps) about 15–30 minutes total
#
# TIP: Run each block in a separate tmux pane or screen session so
# training continues if your SSH connection drops.
#
# Usage: bash run_all.sh

set -e
mkdir -p logs models results tensorboard videos

echo ""
echo "HW3 - Reinforcement Learning - Full Run"

# SPACE INVADERS
echo ""
echo "[SpaceInvaders 1/7] Baseline (lr_mid)"
python train_spaceinvaders.py --config lr_mid --timesteps 1000000 --seed 42

echo ""
echo "[SpaceInvaders 2/7] Learning rate - low"
python train_spaceinvaders.py --config lr_low --timesteps 1000000 --seed 42

echo ""
echo "[SpaceInvaders 3/7] Learning rate - high"
python train_spaceinvaders.py --config lr_high --timesteps 1000000 --seed 42

echo ""
echo "[SpaceInvaders 4/7] Buffer size - small"
python train_spaceinvaders.py --config buf_small --timesteps 1000000 --seed 42

echo ""
echo "[SpaceInvaders 5/7] Buffer size - large"
python train_spaceinvaders.py --config buf_large --timesteps 1000000 --seed 42

echo ""
echo "[SpaceInvaders 6/7] Exploration - low epsilon"
python train_spaceinvaders.py --config eps_low --timesteps 1000000 --seed 42

echo ""
echo "[SpaceInvaders 7/7] Exploration - high epsilon"
python train_spaceinvaders.py --config eps_high --timesteps 1000000 --seed 42

# MOUNTAIN CAR
echo ""
echo "[MountainCar 1/3] DQN - no reward shaping"
python train_mountaincar.py --config dqn_no_shaping --timesteps 200000 --seed 42

echo ""
echo "[MountainCar 2/3] DQN + reward shaping"
python train_mountaincar.py --config dqn_with_shaping --timesteps 200000 --seed 42

echo ""
echo "[MountainCar 3/3] PPO on continuous env"
python train_mountaincar.py --config ppo_continuous --timesteps 200000 --seed 42

# PLOTS & TABLES
echo ""
echo "Generating all plots and summary tables ..."
python evaluate_and_plot.py

# RECORD VIDEOS
echo ""
echo "Recording agent videos ..."
python record_agent.py --env spaceinvaders --config lr_mid    --steps 3000
python record_agent.py --env mountaincar   --config dqn_with_shaping --steps 500
python record_agent.py --env mountaincar   --config ppo_continuous   --steps 500

echo ""
echo "All done. Check:"
echo "  results/     plots + CSV tables"
echo "  models/      saved model checkpoints"
echo "  videos/      MP4 agent recordings"
echo "  tensorboard/ TensorBoard logs"
echo ""
echo "To view TensorBoard: tensorboard --logdir tensorboard/"
