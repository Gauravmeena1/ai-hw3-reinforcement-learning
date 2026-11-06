# evaluate_and_plot.py
# HW3 - Generate all plots and summary tables from saved evaluation logs.
# Run AFTER training is complete:  python evaluate_and_plot.py
#
# Outputs (saved to results/):
#   spaceinvaders_lr.png        - learning-rate ablation
#   spaceinvaders_buffer.png    - buffer-size ablation
#   spaceinvaders_epsilon.png   - exploration ablation
#   mountaincar_curves.png      - learning curves for all 3 MC configs
#   mountaincar_barplot.png     - final mean reward bar chart
#   summary_spaceinvaders.csv   - numeric table for report
#   summary_mountaincar.csv     - numeric table for report

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from train_mountaincar import _make_discrete_env, _make_continuous_env

# Style
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "lines.linewidth": 2,
})
COLORS = {
    "blue":   "#1976D2",
    "green":  "#388E3C",
    "red":    "#D32F2F",
    "orange": "#F57C00",
    "purple": "#7B1FA2",
    "teal":   "#00796B",
    "pink":   "#C2185B",
}


# Helpers

def load_eval_log(log_dir: str):
    """Load evaluations.npz written by EvalCallback.

    Returns (timesteps, mean_rewards, std_rewards) or (None, None, None).
    """
    path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(path):
        print(f"  [warn] Not found: {path}")
        return None, None, None
    data = np.load(path)
    ts   = data["timesteps"]
    res  = data["results"]          # shape: (n_evals, n_eval_episodes)
    return ts, res.mean(axis=1), res.std(axis=1)


def smooth(arr: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def _plot_lines(ax, entries, x_scale=1e6, x_label="Timesteps (×10⁶)", y_label="Mean Episode Reward"):
    """Plot multiple learning curves on ax.

    entries: list of (log_dir, label, color)
    """
    any_data = False
    for log_dir, label, color in entries:
        ts, mean, std = load_eval_log(log_dir)
        if ts is None:
            continue
        any_data = True
        xs = ts / x_scale
        ms = smooth(mean)
        ax.plot(xs, ms, label=label, color=color)
        ax.fill_between(xs, ms - std, ms + std, alpha=0.15, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if any_data:
        ax.legend(fontsize=9)
    return any_data


# SpaceInvaders plots

def plot_spaceinvaders_lr():
    fig, ax = plt.subplots(figsize=(6, 4))
    entries = [
        ("logs/spaceinvaders/lr_low",  "LR = 1×10⁻⁴",          COLORS["blue"]),
        ("logs/spaceinvaders/lr_mid",  "LR = 5×10⁻⁴ (baseline)",COLORS["green"]),
        ("logs/spaceinvaders/lr_high", "LR = 1×10⁻³",           COLORS["red"]),
    ]
    _plot_lines(ax, entries)
    ax.set_title("SpaceInvaders — Effect of Learning Rate")
    fig.tight_layout()
    fig.savefig("results/spaceinvaders_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/spaceinvaders_lr.png")


def plot_spaceinvaders_buffer():
    fig, ax = plt.subplots(figsize=(6, 4))
    entries = [
        ("logs/spaceinvaders/buf_small", "Buffer = 50 K",          COLORS["orange"]),
        ("logs/spaceinvaders/lr_mid",    "Buffer = 100 K (baseline)",COLORS["green"]),
        ("logs/spaceinvaders/buf_large", "Buffer = 200 K",          COLORS["purple"]),
    ]
    _plot_lines(ax, entries)
    ax.set_title("SpaceInvaders — Effect of Replay Buffer Size")
    fig.tight_layout()
    fig.savefig("results/spaceinvaders_buffer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/spaceinvaders_buffer.png")


def plot_spaceinvaders_epsilon():
    fig, ax = plt.subplots(figsize=(6, 4))
    entries = [
        ("logs/spaceinvaders/eps_low",  "ε_end = 0.005",         COLORS["teal"]),
        ("logs/spaceinvaders/lr_mid",   "ε_end = 0.01 (baseline)",COLORS["green"]),
        ("logs/spaceinvaders/eps_high", "ε_end = 0.10",          COLORS["pink"]),
    ]
    _plot_lines(ax, entries)
    ax.set_title("SpaceInvaders — Effect of Final Exploration ε")
    fig.tight_layout()
    fig.savefig("results/spaceinvaders_epsilon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/spaceinvaders_epsilon.png")


# MountainCar plots

def plot_mountaincar_curves():
    fig, ax = plt.subplots(figsize=(7, 4))
    entries = [
        ("logs/mountaincar/dqn_no_shaping",  "DQN — no shaping",   COLORS["red"]),
        ("logs/mountaincar/dqn_with_shaping","DQN + reward shaping",COLORS["green"]),
        ("logs/mountaincar/ppo_continuous",  "PPO — continuous",    COLORS["blue"]),
    ]
    _plot_lines(ax, entries, x_scale=1e3, x_label="Timesteps (×10³)")
    ax.set_title("MountainCar — Learning Curves (3 Conditions)")
    fig.tight_layout()
    fig.savefig("results/mountaincar_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/mountaincar_curves.png")


def plot_mountaincar_barplot():
    """Load best saved models, evaluate 50 episodes each, draw bar chart."""
    configs = [
        ("dqn_no_shaping",   "DQN\n(no shaping)",    COLORS["red"],   "dqn", False),
        ("dqn_with_shaping", "DQN\n(reward shaping)", COLORS["green"], "dqn", True),
        ("ppo_continuous",   "PPO\n(continuous)",     COLORS["blue"],  "ppo", False),
    ]

    means, stds, labels, colors = [], [], [], []
    for cfg_name, label, color, method, shaping in configs:
        model_path = f"models/mountaincar/{cfg_name}/best_model"
        if not os.path.exists(model_path + ".zip"):
            model_path = f"models/mountaincar/{cfg_name}/final_model"
        if not os.path.exists(model_path + ".zip"):
            print(f"  [warn] No model found for {cfg_name}, skipping bar.")
            continue

        if method == "ppo":
            model = PPO.load(model_path)
            env   = DummyVecEnv([_make_continuous_env(seed=999)])
        else:
            model = DQN.load(model_path)
            env   = DummyVecEnv([_make_discrete_env(reward_shaping=False, seed=999)])

        m, s = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
        means.append(m); stds.append(s); labels.append(label); colors.append(color)
        env.close()
        print(f"  {cfg_name}: mean={m:.1f} ± {s:.1f}")

    if not means:
        print("  No model data available for bar plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Episode Reward (50 episodes)")
    ax.set_title("MountainCar — Final Agent Performance")
    fig.tight_layout()
    fig.savefig("results/mountaincar_barplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/mountaincar_barplot.png")


# Summary tables

def _collect_row(log_dir: str) -> dict:
    ts, mean, std = load_eval_log(log_dir)
    if ts is None:
        return {"Final Mean": "—", "Final Std": "—", "Best Mean": "—", "Steps": "—"}
    return {
        "Final Mean": f"{mean[-1]:.1f}",
        "Final Std":  f"{std[-1]:.1f}",
        "Best Mean":  f"{mean.max():.1f}",
        "Steps":      f"{int(ts[-1]):,}",
    }


def build_summary_tables():
    # SpaceInvaders
    si_rows = {
        "LR=1e-4":             _collect_row("logs/spaceinvaders/lr_low"),
        "LR=5e-4 (baseline)":  _collect_row("logs/spaceinvaders/lr_mid"),
        "LR=1e-3":             _collect_row("logs/spaceinvaders/lr_high"),
        "Buffer=50K":          _collect_row("logs/spaceinvaders/buf_small"),
        "Buffer=100K":         _collect_row("logs/spaceinvaders/lr_mid"),
        "Buffer=200K":         _collect_row("logs/spaceinvaders/buf_large"),
        "ε_end=0.005":         _collect_row("logs/spaceinvaders/eps_low"),
        "ε_end=0.01":          _collect_row("logs/spaceinvaders/lr_mid"),
        "ε_end=0.10":          _collect_row("logs/spaceinvaders/eps_high"),
    }
    df_si = pd.DataFrame(si_rows).T
    df_si.to_csv("results/summary_spaceinvaders.csv")
    print("\nSpaceInvaders summary:\n", df_si.to_string())

    # MountainCar
    mc_rows = {
        "DQN (no shaping)":    _collect_row("logs/mountaincar/dqn_no_shaping"),
        "DQN (reward shaping)":_collect_row("logs/mountaincar/dqn_with_shaping"),
        "PPO (continuous)":    _collect_row("logs/mountaincar/ppo_continuous"),
    }
    df_mc = pd.DataFrame(mc_rows).T
    df_mc.to_csv("results/summary_mountaincar.csv")
    print("\nMountainCar summary:\n", df_mc.to_string())

    return df_si, df_mc


# Main

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("\n[1/6] SpaceInvaders: Learning rate plot ...")
    plot_spaceinvaders_lr()

    print("\n[2/6] SpaceInvaders: Buffer size plot ...")
    plot_spaceinvaders_buffer()

    print("\n[3/6] SpaceInvaders: Exploration plot ...")
    plot_spaceinvaders_epsilon()

    print("\n[4/6] MountainCar: Learning curves ...")
    plot_mountaincar_curves()

    print("\n[5/6] MountainCar: Bar chart (loads models — may take ~1 min) ...")
    plot_mountaincar_barplot()

    print("\n[6/6] Summary tables ...")
    build_summary_tables()

    print("\n=== Done! All outputs saved to results/ ===")
