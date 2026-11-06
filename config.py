# config.py - All hyperparameter configurations for HW3 experiments

# ============================================================
#  SPACE INVADERS - DQN Experiment Configurations
# ============================================================
# Three ablation groups:
#   Group A: Learning rate      (lr_low / lr_mid / lr_high)
#   Group B: Replay buffer size (buf_small / lr_mid / buf_large)
#   Group C: Final exploration  (eps_low / lr_mid / eps_high)
#
# "lr_mid" is the shared baseline across all groups.

SPACEINVADERS_CONFIGS = {
    # Group A: Learning Rate
    "lr_low": {
        "learning_rate":        1e-4,
        "buffer_size":          100_000,
        "exploration_final_eps": 0.01,
    },
    "lr_mid": {                          # baseline
        "learning_rate":        5e-4,
        "buffer_size":          100_000,
        "exploration_final_eps": 0.01,
    },
    "lr_high": {
        "learning_rate":        1e-3,
        "buffer_size":          100_000,
        "exploration_final_eps": 0.01,
    },

    # Group B: Replay Buffer Size
    "buf_small": {
        "learning_rate":        5e-4,
        "buffer_size":          50_000,
        "exploration_final_eps": 0.01,
    },
    "buf_large": {
        "learning_rate":        5e-4,
        "buffer_size":          200_000,
        "exploration_final_eps": 0.01,
    },

    # Group C: Final Exploration ε
    "eps_low": {
        "learning_rate":        5e-4,
        "buffer_size":          100_000,
        "exploration_final_eps": 0.005,
    },
    "eps_high": {
        "learning_rate":        5e-4,
        "buffer_size":          100_000,
        "exploration_final_eps": 0.10,
    },
}

# Shared DQN hyperparameters for SpaceInvaders (fixed across all configs)
SPACEINVADERS_FIXED = {
    "learning_starts":          100_000,   # warm-up steps before learning
    "batch_size":               32,
    "gamma":                    0.99,      # discount factor
    "train_freq":               4,         # update every 4 env steps
    "gradient_steps":           1,
    "target_update_interval":   1_000,     # hard update target network
    "exploration_fraction":     0.1,       # fraction of training for ε decay
    "exploration_initial_eps":  1.0,
    "n_stack":                  4,         # frame stacking
}

SPACEINVADERS_TOTAL_TIMESTEPS = 1_000_000   # ~1 hr on RTX 3080


# ============================================================
#  MOUNTAIN CAR - Experiment Configurations
# ============================================================
# Three conditions:
#   1. DQN on MountainCar-v0 without reward shaping  -> baseline (likely fails)
#   2. DQN on MountainCar-v0 with reward shaping     -> key contribution
#   3. PPO on MountainCarContinuous-v0               -> algorithm comparison

MOUNTAINCAR_CONFIGS = {
    "dqn_no_shaping": {
        "method":           "dqn",
        "env_id":           "MountainCar-v0",
        "reward_shaping":   False,
        "learning_rate":    1e-3,
    },
    "dqn_with_shaping": {
        "method":           "dqn",
        "env_id":           "MountainCar-v0",
        "reward_shaping":   True,
        "learning_rate":    1e-3,
    },
    "ppo_continuous": {
        "method":           "ppo",
        "env_id":           "MountainCarContinuous-v0",
        "reward_shaping":   False,
        "learning_rate":    3e-4,
    },
}

# Shared DQN hyperparameters for MountainCar-v0
MOUNTAINCAR_DQN_FIXED = {
    "buffer_size":              50_000,
    "learning_starts":          1_000,
    "batch_size":               64,
    "gamma":                    0.99,
    "train_freq":               4,
    "gradient_steps":           1,
    "target_update_interval":   500,
    "exploration_fraction":     0.3,
    "exploration_initial_eps":  1.0,
    "exploration_final_eps":    0.05,
}

# Shared PPO hyperparameters for MountainCarContinuous-v0
MOUNTAINCAR_PPO_FIXED = {
    "n_steps":       2048,
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.0,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
}

MOUNTAINCAR_TOTAL_TIMESTEPS = 200_000   # ~5 min per config on CPU
