#!/usr/bin/env python3
"""Baseline episode loop: random and rule-based agents over a full shift.

This is the "example training loop" deliverable in its simplest honest form -
it runs the environment, accumulates return, and prints the KPI panel that the
research question actually cares about. ``scripts/train_ppo.py`` shows the same
loop driven by a learned policy instead.

Usage:
    python scripts/run_baseline.py --episodes 20 --agent rule_based
    python scripts/run_baseline.py --agent random --no-telemetry
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ward_cgm_sim.agents import RandomAgent, RuleBasedAgent  # noqa: E402
from ward_cgm_sim.config import SimConfig  # noqa: E402
from ward_cgm_sim.core.engine import WardEngine  # noqa: E402

AGENTS = {"random": RandomAgent, "rule_based": RuleBasedAgent}


def run_episode(agent, cfg: SimConfig, seed: int) -> dict:
    """Run one 12-hour shift and return its KPI dict plus the return."""
    engine = WardEngine(cfg, seed=seed)
    agent.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.act(engine)
        _obs, reward, terminated, truncated, info = engine.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    result = dict(info["kpi"])
    result["total_reward"] = total_reward
    result["steps_survived"] = steps
    result["termination_reason"] = info["termination_reason"]
    result["reward_components"] = info["reward_components"]
    return result


def summarise(results: list[dict], label: str) -> None:
    def mean(key):
        values = [r[key] for r in results if r.get(key) is not None]
        return statistics.mean(values) if values else float("nan")

    incident_free = sum(1 for r in results if r.get("incident_free_shift"))
    print(f"\n=== {label} over {len(results)} shifts ===")
    print(f"  return                    {mean('total_reward'):+8.1f}")
    print(f"  steps survived            {mean('steps_survived'):8.1f} / {results[0]['steps'] and 144}")
    print(f"  INCIDENT-FREE SHIFTS      {incident_free:8d} / {len(results)}"
          f"  ({100 * incident_free / len(results):.0f}%)")
    print(f"  serious adverse events    {mean('serious_adverse_events'):8.2f}")
    print(f"  time below range (steps)  {mean('time_below_range_steps'):8.1f}")
    print(f"  severe hypo events        {mean('severe_hypo_events'):8.2f}")
    print(f"  alarms raised             {mean('alarms_raised'):8.1f}")
    print(f"  false alarm rate          {mean('false_alarm_rate'):8.2f}")
    print(f"  mean alarm response (steps){mean('mean_alarm_response_steps'):7.2f}")
    print(f"  PoC tests                 {mean('poc_tests'):8.1f}")
    print(f"  treatments w/o PoC        {mean('treatments_without_poc'):8.1f}")
    print(f"  correct enrolments        {mean('correct_enrolments'):8.2f}")
    print(f"  incorrect enrolments      {mean('incorrect_enrolments'):8.2f}")
    print(f"  missed eligible           {mean('missed_eligible'):8.2f}")
    print(f"  correct de-enrolments     {mean('correct_deenrolments'):8.2f}")
    print(f"  signal loss ignored steps {mean('signal_loss_ignored_steps'):8.1f}")
    print(f"  discharges                {mean('discharges'):8.2f}")
    print(f"  discharge delay steps     {mean('discharge_delay_steps'):8.1f}")
    print(f"  max queue length          {mean('max_queue_length'):8.1f}")
    print(f"  overcrowding steps        {mean('overcrowding_steps'):8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(AGENTS), default="rule_based")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="run the CGM-off counterfactual (no dashboard, no alarms)",
    )
    parser.add_argument("--components", action="store_true", help="print the reward breakdown")
    args = parser.parse_args()

    cfg = SimConfig()
    cfg.telemetry_enabled = not args.no_telemetry

    agent = AGENTS[args.agent](seed=args.seed)
    results = [run_episode(agent, cfg, seed=args.seed + i) for i in range(args.episodes)]

    label = f"{args.agent} / telemetry {'ON' if cfg.telemetry_enabled else 'OFF'}"
    summarise(results, label)

    if args.components:
        print("\n  reward components (mean per shift):")
        keys = sorted({k for r in results for k in r["reward_components"]})
        for key in keys:
            values = [r["reward_components"].get(key, 0.0) for r in results]
            print(f"    {key:36s} {statistics.mean(values):+8.2f}")


if __name__ == "__main__":
    main()
