#!/bin/bash
# setup.sh - Install all dependencies for HW3 RL project
# Run once before training: bash setup.sh

set -e

echo "=== Step 1: Install Python dependencies ==="
pip install -r requirements.txt

echo "=== Step 2: Install Atari ROMs ==="
# AutoROM automatically downloads and installs the free Atari ROMs
pip install autorom[accept-rom-license]
AutoROM --accept-license

echo "=== Step 3: Verify Gymnasium Atari ==="
python -c "
import gymnasium as gym
env = gym.make('ALE/SpaceInvaders-v5')
obs, _ = env.reset()
print(f'SpaceInvaders OK — obs shape: {obs.shape}')
env.close()

env2 = gym.make('MountainCar-v0')
obs2, _ = env2.reset()
print(f'MountainCar OK — obs shape: {obs2.shape}')
env2.close()

env3 = gym.make('MountainCarContinuous-v0')
obs3, _ = env3.reset()
print(f'MountainCarContinuous OK — obs shape: {obs3.shape}')
env3.close()
"

echo "=== Step 4: Check CUDA ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "=== Setup complete! You can now run: bash run_all.sh ==="
