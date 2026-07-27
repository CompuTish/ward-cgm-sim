#!/usr/bin/env python3
"""The core experiment: does CGM telemetry improve ward workflow and safety?

The comparison is *telemetry versus routine monitoring*, not telemetry versus
nothing. With ``telemetry_enabled=False`` the dashboard and every alarm
disappear, and deteriorating patients are found the way they are found on a
ward today: routine capillary rounds, symptom recognition, and the nurse
happening to check. That baseline is modelled explicitly in
``config.UsualCareConfig``.

Both arms are run on *matched seeds*, so the same patients, the same glucose
trajectories and the same admissions occur in each arm. Any difference in the
outcome measures is therefore attributable to the monitoring strategy and the
policy's response to it, not to sampling noise between arms.

Primary outcome
    mean hypoglycaemia detection delay - how long a patient spends below
    3.9 mmol/L before anybody knows about it.

Secondary outcomes
    time below range, severe hypoglycaemia events, serious adverse events,
    incident-free shift rate, alarm burden and false-alarm rate, enrolment
    precision and recall, discharge delay, queue length and overcrowding.

This is a simulation study of a workflow model. It cannot demonstrate clinical
benefit; it can only show whether a modelled mechanism plausibly produces one,
and how sensitive that is to the parameters in ``config.py``.

Usage:
    python scripts/evaluate.py --episodes 40
    python scripts/evaluate.py --episodes 40 --agent random
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ward_cgm_sim.agents import RandomAgent, RuleBasedAgent  # noqa: E402
from ward_cgm_sim.config import SimConfig  # noqa: E402
from ward_cgm_sim.core.engine import WardEngine  # noqa: E402

AGENTS = {"random": RandomAgent, "rule_based": RuleBasedAgent}

# (kpi key, label, lower_is_better)
#
# The PRIMARY outcomes are the cohort-restricted ones. The monitored cohort is
# fixed at handover and selected identically in both arms, so it is the same
# set of patients either way - the only like-for-like comparison available.
# Ward-wide figures are reported underneath as context; they are diluted by the
# large majority of patients who are never eligible for telemetry at all, and
# so understate any effect.
OUTCOMES = [
    ("cohort_time_below_range_steps", "cohort time below range (patient-steps)", True),
    ("cohort_severe_hypo_events", "cohort severe hypo events", True),
    ("cohort_hypo_episodes", "cohort hypo episodes (denominator)", None),
    ("time_below_range_steps", "time below range (patient-steps)", True),
    ("severe_hypo_events", "severe hypo events", True),
    ("serious_adverse_events", "serious adverse events", True),
    ("incident_free_shift", "incident-free shifts (rate)", False),
    ("alarms_raised", "alarms raised", None),
    ("false_alarm_rate", "false alarm rate", True),
    ("mean_alarm_response_steps", "alarm response (steps)", True),
    ("poc_tests", "point-of-care tests", None),
    ("correct_enrolments", "correct enrolments", False),
    ("incorrect_enrolments", "incorrect enrolments", True),
    ("missed_eligible", "missed eligible patients", True),
    ("discharges", "discharges", False),
    ("discharge_delay_steps", "discharge delay (patient-steps)", True),
    ("max_queue_length", "peak admissions queue", True),
    ("overcrowding_steps", "overcrowding (steps)", True),
]


def run_episode(agent_cls, cfg: SimConfig, seed: int) -> dict:
    engine = WardEngine(cfg, seed=seed)
    agent = agent_cls(seed=seed)
    agent.reset()
    total = 0.0
    while True:
        _obs, reward, terminated, truncated, info = engine.step(agent.act(engine))
        total += reward
        if terminated or truncated:
            break
    kpi = dict(info["kpi"])
    kpi["total_reward"] = total
    return kpi


def mean_of(results: list[dict], key: str) -> float | None:
    values = []
    for r in results:
        v = r.get(key)
        if v is None:
            continue
        values.append(float(v))
    return statistics.mean(values) if values else None


def bootstrap_ci(
    results: list[dict],
    numerator: str,
    denominator: str,
    draws: int = 2000,
    seed: int = 12345,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI, resampling whole SHIFTS.

    Shifts are the unit of randomisation, and events within a shift are
    correlated (same ward, same staff, same patients), so resampling
    individual events would understate the uncertainty. With ~30 events per
    arm the interval is wide, which is the point: a point estimate alone
    invites over-reading a very small sample.
    """
    if not results:
        return None
    rng = random.Random(seed)
    n = len(results)
    estimates = []
    for _ in range(draws):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        total_n = sum(r.get(numerator) or 0 for r in sample)
        total_d = sum(r.get(denominator) or 0 for r in sample)
        if total_d:
            estimates.append(total_n / total_d)
    if len(estimates) < draws * 0.5:
        return None
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return lo, hi


def pooled_ratio(results: list[dict], numerator: str, denominator: str) -> float | None:
    """Event-level pooling: sum numerators, sum denominators, then divide.

    Averaging per-shift ratios and discarding shifts with a zero denominator
    silently compares different subsets of shifts between the two arms - the
    telemetry arm had far more shifts with a detected episode, so the
    macro-average was computed over a different (and better) set of shifts than
    the comparator's. Pooling uses every episode in both arms.
    """
    total_n = sum(r.get(numerator) or 0 for r in results)
    total_d = sum(r.get(denominator) or 0 for r in results)
    return total_n / total_d if total_d else None


def fmt(value: float | None) -> str:
    return "     n/a" if value is None else f"{value:8.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(AGENTS), default="rule_based")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent_cls = AGENTS[args.agent]
    seeds = [args.seed + i for i in range(args.episodes)]

    on_cfg = SimConfig()
    on_cfg.telemetry_enabled = True
    off_cfg = SimConfig()
    off_cfg.telemetry_enabled = False

    # Matched seeds: identical patients and trajectories in both arms.
    telemetry_on = [run_episode(agent_cls, on_cfg, s) for s in seeds]
    telemetry_off = [run_episode(agent_cls, off_cfg, s) for s in seeds]

    print(f"\nCGM telemetry vs routine monitoring - {args.agent} policy, "
          f"{args.episodes} matched shifts, seed {args.seed}")
    print("(academic simulation model; not evidence of clinical benefit)\n")

    # ---- Primary outcomes, pooled at the event level ----------------------
    print("PRIMARY OUTCOMES (monitored cohort, pooled over all episodes)")
    print("95% intervals are a percentile bootstrap resampling whole shifts.")
    print(f"{'  outcome':38s} {'telemetry':>20s} {'routine':>20s}")
    print("  " + "-" * 76)

    def with_ci(results, num, den) -> str:
        point = pooled_ratio(results, num, den)
        if point is None:
            return "     n/a"
        ci = bootstrap_ci(results, num, den)
        if ci is None:
            return f"{point:8.2f}"
        return f"{point:6.2f} [{ci[0]:.2f},{ci[1]:.2f}]"

    for label, num, den in (
        ("detection rate", "cohort_hypo_detections", "cohort_hypo_episodes"),
        (
            "detection delay | detected (steps)",
            "cohort_detection_delay_steps_total",
            "cohort_hypo_detections",
        ),
    ):
        print(f"  {label:36s} {with_ci(telemetry_on, num, den):>20s} "
              f"{with_ci(telemetry_off, num, den):>20s}")

    on_ep = sum(r.get("cohort_hypo_episodes") or 0 for r in telemetry_on)
    off_ep = sum(r.get("cohort_hypo_episodes") or 0 for r in telemetry_off)
    on_det = sum(r.get("cohort_hypo_detections") or 0 for r in telemetry_on)
    off_det = sum(r.get("cohort_hypo_detections") or 0 for r in telemetry_off)
    print(f"  {'episodes / detected (counts)':36s} "
          f"{on_ep:9d}/{on_det:<10d} {off_ep:9d}/{off_det:<10d}")
    print(f"\n  WARD-WIDE, same pooling (all patients, not just the cohort):")
    for label, num, den in (
        ("  detection rate", "hypo_detections", "hypo_episodes"),
        (
            "  detection delay | detected (steps)",
            "hypo_detection_delay_steps_total",
            "hypo_detections",
        ),
    ):
        print(f"  {label:36s} {with_ci(telemetry_on, num, den):>20s} "
              f"{with_ci(telemetry_off, num, den):>20s}")
    print(
        "\n  Delay is CONDITIONAL ON DETECTION and is therefore censored: an\n"
        "  episode nobody ever found contributes no delay at all. Read it\n"
        "  together with the detection rate, never on its own - an arm that\n"
        "  detects only the most obvious events will look deceptively fast.\n"
    )

    print("SECONDARY OUTCOMES (per-shift means)")
    print(f"{'  outcome':34s} {'telemetry':>9s} {'routine':>9s} {'difference':>11s}")
    print("  " + "-" * 64)

    for key, label, lower_better in OUTCOMES:
        on_value = mean_of(telemetry_on, key)
        off_value = mean_of(telemetry_off, key)
        if on_value is None and off_value is None:
            continue
        if on_value is None or off_value is None:
            diff = "        -"
        else:
            delta = on_value - off_value
            marker = ""
            if lower_better is not None and abs(delta) > 1e-9:
                improved = delta < 0 if lower_better else delta > 0
                marker = " +" if improved else " -"
            diff = f"{delta:+9.2f}{marker}"
        print(f"  {label:34s} {fmt(on_value)} {fmt(off_value)} {diff:>11s}")

    print("\n  '+' marks the arm-difference favouring telemetry on that outcome.")
    print("  Detection delay and detection rate are the primary outcomes; the rest")
    print("  describe alarm burden, enrolment quality and ward flow. Every number")
    print("  is a function of the parameters in ward_cgm_sim/config.py, so treat")
    print("  this as a sensitivity exercise rather than a result.\n")


if __name__ == "__main__":
    main()
