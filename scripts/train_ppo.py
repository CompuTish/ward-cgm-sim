#!/usr/bin/env python3
"""Worked example: train a PPO policy on the ward environment.

This is an example, not a result. A policy trained for a few hundred thousand
steps on a 296-dimensional partially observable problem will not be good; the
point is to show the environment plugs into a standard RL stack unmodified, and
to give a starting point for real experiments.

Needs the training extras:  pip install -e ".[train]"

Usage:
    python scripts/train_ppo.py --timesteps 200000
    python scripts/train_ppo.py --timesteps 200000 --no-telemetry
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover - guarded import by design
    sys.exit(
        "stable-baselines3 is not installed.\n"
        'Install the training extras:  pip install -e ".[train]"'
    )

from ward_cgm_sim.agents import RuleBasedAgent  # noqa: E402
from ward_cgm_sim.config import SimConfig  # noqa: E402
from ward_cgm_sim.core.engine import WardEngine  # noqa: E402
from ward_cgm_sim.env import WardCGMTelemetryEnv  # noqa: E402


def make(telemetry: bool):
    def factory():
        cfg = SimConfig()
        cfg.telemetry_enabled = telemetry
        return Monitor(WardCGMTelemetryEnv(config=cfg))

    return factory


def rule_based_reference(telemetry: bool, episodes: int, seed: int) -> float:
    """What the hand-written policy scores, so the learned one has a bar."""
    cfg = SimConfig()
    cfg.telemetry_enabled = telemetry
    agent = RuleBasedAgent()
    total = 0.0
    for index in range(episodes):
        engine = WardEngine(cfg, seed=seed + index)
        agent.reset()
        while True:
            _o, reward, terminated, truncated, _i = engine.step(agent.act(engine))
            total += reward
            if terminated or truncated:
                break
    return total / episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save", default="ppo_ward.zip")
    args = parser.parse_args()

    telemetry = not args.no_telemetry

    baseline = rule_based_reference(telemetry, args.eval_episodes, args.seed + 10_000)
    print(f"rule-based reference: {baseline:+.1f} mean return over "
          f"{args.eval_episodes} shifts")

    env = DummyVecEnv([make(telemetry)])
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        gamma=1.0,  # finite horizon; see docs/POMDP.md
        ent_coef=0.01,  # the action space is large and mostly context-invalid
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f"saved {args.save}")

    # Evaluate on held-out seeds.
    eval_env = WardCGMTelemetryEnv(config=env.envs[0].unwrapped.config)
    returns = []
    for index in range(args.eval_episodes):
        obs, _info = eval_env.reset(seed=args.seed + 10_000 + index)
        total = 0.0
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    eval_env.close()

    learned = sum(returns) / len(returns)
    print(f"\nlearned policy:       {learned:+.1f} mean return")
    print(f"rule-based reference: {baseline:+.1f}")
    print(
        "\nA learned policy below the reference is the expected outcome at this "
        "budget - the hand-written nurse encodes a lot of prior knowledge."
    )


if __name__ == "__main__":
    main()
