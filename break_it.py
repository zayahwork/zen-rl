"""
break_it.py — generate labeled RL training runs for ZEN's eval set.

Usage:
    python break_it.py good        # baseline only
    python break_it.py all         # everything (~45 min, walk away)
    python break_it.py list        # show configs without running

Logs land in evals/runs/<name>/
View:  tensorboard --logdir evals/runs
"""

import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

LOGDIR = Path("evals/runs")
STEPS = 50_000

# The dict key IS the ground-truth label.
# The comment is a HYPOTHESIS, not a fact. Your job tomorrow is to
# confirm or reject each one by looking at what actually happened.
# Some of these will not fail. That is useful information too.
CONFIGS = {
    # --- baseline ---
    "good": dict(),

    # --- learning rate family ---
    "lr_too_high":      dict(learning_rate=1e-1),   # CONFIRMED: kl spike -> collapse
    "lr_slightly_high": dict(learning_rate=3e-3),   # ? borderline — recovers or not?
    "lr_too_low":       dict(learning_rate=1e-6),   # ? flat reward, never moves

    # --- normalization ---
    "no_norm": dict(normalize_advantage=False),     # CONFIRMED: |pg_loss| ~10

    # --- entropy ---
    "high_entropy": dict(ent_coef=0.5),             # ? never commits, stays random

    # --- rollout / batch shape ---
    "tiny_rollout":   dict(n_steps=32),             # ? bad value estimates
    "huge_rollout":   dict(n_steps=8192),           # ? few updates, slow
    "batch_gt_steps": dict(n_steps=64, batch_size=256),  # ? batch bigger than rollout

    # --- discounting ---
    "gamma_too_low":   dict(gamma=0.5),             # ? myopic, can't see pole falling
    "gamma_is_one":    dict(gamma=1.0),             # ? no discount, value may blow up
    "gae_lambda_zero": dict(gae_lambda=0.0),        # ? pure TD, high bias

    # --- clipping ---
    "clip_huge": dict(clip_range=10.0),             # ? clip does nothing = vanilla PG
    "clip_tiny": dict(clip_range=0.001),            # ? updates too small to learn

    # --- epochs ---
    "epochs_many": dict(n_epochs=50),               # ? overfits each batch, kl drifts
    "epochs_one":  dict(n_epochs=1),                # ? sample inefficient, slow
}

DEFAULTS = dict(
    learning_rate=3e-4,
    batch_size=64,
    normalize_advantage=True,
    n_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    n_epochs=10,
)


def run(name: str) -> None:
    cfg = {**DEFAULTS, **CONFIGS[name]}

    changed = {k: v for k, v in CONFIGS[name].items()}
    print(f"\n{'=' * 60}")
    print(f"=== {name} ===")
    print(f"    changed: {changed or 'nothing (baseline)'}")

    env = gym.make("CartPole-v1")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=0,
        tensorboard_log=str(LOGDIR),
        **cfg,
    )
    model.learn(total_timesteps=STEPS, tb_log_name=name)
    env.close()
    print(f"    -> evals/runs/{name}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    LOGDIR.mkdir(parents=True, exist_ok=True)
    arg = sys.argv[1]

    if arg == "list":
        for k, v in CONFIGS.items():
            print(f"  {k:18} {v or '(baseline)'}")
        return

    if arg == "all":
        total = len(CONFIGS)
        for i, name in enumerate(CONFIGS, 1):
            print(f"\n[{i}/{total}]", end="")
            try:
                run(name)
            except Exception as e:
                print(f"    FAILED TO RUN: {e}")
        print(f"\n{'=' * 60}\nDone. {total} runs in evals/runs/")
        print("Next: tensorboard --logdir evals/runs")
        return

    if arg in CONFIGS:
        run(arg)
        return

    print(f"unknown config: {arg}")
    print(f"choices: {', '.join(CONFIGS)}, all, list")
    sys.exit(1)


if __name__ == "__main__":
    main()
