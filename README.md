# Reinforcement Learning (HW3)

Project 3 for Artificial Intelligence (NYCU, Spring 2026). DQN on Atari Space Invaders with hyperparameter comparisons, and DQN or PPO on MountainCar with and without reward shaping. Built with Stable-Baselines3 and Gymnasium.

## Files

| File | Purpose |
|---|---|
| `config.py` | Hyperparameters for every experiment |
| `train_spaceinvaders.py` | DQN on `ALE/SpaceInvaders-v5` |
| `train_mountaincar.py` | DQN on `MountainCar-v0`, PPO on `MountainCarContinuous-v0`, reward shaping wrapper |
| `evaluate_and_plot.py` | Plots and summary tables from the saved logs |
| `record_agent.py` | MP4 videos of trained agents |
| `setup.sh`, `run_all.sh` | Install, then run everything |

## Run

```bash
bash setup.sh
bash run_all.sh
```

One experiment at a time:

```bash
python train_spaceinvaders.py --config lr_mid --timesteps 1000000
python train_mountaincar.py --config dqn_with_shaping --timesteps 200000
python evaluate_and_plot.py
```

Outputs go to `logs/`, `models/`, `results/`, `tensorboard/` and `videos/`, which are not in Git. Training all ten configs takes about 7 to 10 hours on a GPU.

## Results

Space Invaders, 1M steps, one seed, 5 evaluation episodes at the last checkpoint:

| Config | Final mean | Std | Best mean |
|---|---|---:|---:|
| LR 1e-4 (stopped at 650K steps) | 444.0 | 117.2 | 556.0 |
| LR 5e-4, buffer 100K, eps 0.01 (baseline) | 427.0 | 90.0 | 534.0 |
| Buffer 50K | 389.0 | 159.6 | 622.0 |
| Buffer 200K | 600.0 | 205.4 | 657.0 |
| Final eps 0.005 | 353.0 | 217.1 | 531.0 |
| Final eps 0.10 | 518.0 | 176.7 | 531.0 |

The LR 1e-3 run was not finished.

MountainCar, 200K steps, one seed, raw environment reward at the last checkpoint:

| Condition | Final mean | Std | Best mean |
|---|---|---:|---:|
| DQN, no shaping | -200.0 | 0.0 | -143.5 |
| DQN, potential-based shaping | -125.8 | 18.6 | -125.7 |
| PPO, continuous actions | about 0 | 0.0 | about 0 |

The two MountainCar environments score differently, so the PPO row can't be compared directly with the DQN rows. In the continuous version, reaching the goal gives +100, so a score near 0 means the PPO agent did not reach it.
