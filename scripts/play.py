#!/usr/bin/env python3
"""Play the ward shift yourself, natively, with the top-down renderer.

Arrow keys or WASD to walk. Interactions apply to the bed you are standing
next to; dashboard actions need you at the nurse station in the middle.

  D  check the telemetry dashboard      C  check patient
  N  review notes / drug chart          K  ask for verbal consent
  E  enrol                              R  review enrolled eligibility
  X  de-enrol                           SPACE respond to alarm
  G  point-of-care glucose              1  treat hypoglycaemia
  2  treat hyperglycaemia               Q  escalate to medical / diabetes
  F1 ask HCA    F2 ask nurse    F3 ask doctor    F4 ask surgeon
  T  troubleshoot sensor                S  support discharge
  B  prioritise bed flow                .  wait
  TAB hand over to the rule-based policy for one step
  ESC quit

ACADEMIC MODEL - the treatment actions are simplified placeholders, not
clinical guidance.

Usage:
    python scripts/play.py [--seed 0] [--no-telemetry] [--watch]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame  # noqa: E402

from ward_cgm_sim.agents import RuleBasedAgent  # noqa: E402
from ward_cgm_sim.config import SimConfig  # noqa: E402
from ward_cgm_sim.core.actions import Action  # noqa: E402
from ward_cgm_sim.core.engine import WardEngine  # noqa: E402
from ward_cgm_sim.render.pygame_renderer import WardRenderer  # noqa: E402

KEY_ACTIONS = {
    pygame.K_UP: Action.MOVE_UP,
    pygame.K_w: Action.MOVE_UP,
    pygame.K_DOWN: Action.MOVE_DOWN,
    pygame.K_s: Action.MOVE_DOWN,
    pygame.K_LEFT: Action.MOVE_LEFT,
    pygame.K_a: Action.MOVE_LEFT,
    pygame.K_RIGHT: Action.MOVE_RIGHT,
    pygame.K_d: Action.CHECK_DASHBOARD,
    pygame.K_c: Action.CHECK_PATIENT,
    pygame.K_n: Action.REVIEW_NOTES,
    pygame.K_k: Action.ASK_CONSENT,
    pygame.K_e: Action.ENROL,
    pygame.K_r: Action.REVIEW_ELIGIBILITY,
    pygame.K_x: Action.DEENROL,
    pygame.K_SPACE: Action.RESPOND_ALARM,
    pygame.K_g: Action.POC_GLUCOSE_TEST,
    pygame.K_1: Action.TREAT_HYPO,
    pygame.K_2: Action.TREAT_HYPER,
    pygame.K_q: Action.ESCALATE,
    pygame.K_F1: Action.ASK_HELP_HCA,
    pygame.K_F2: Action.ASK_HELP_NURSE,
    pygame.K_F3: Action.ASK_HELP_DOCTOR,
    pygame.K_F4: Action.ASK_HELP_SURGEON,
    pygame.K_t: Action.TROUBLESHOOT_SENSOR,
    pygame.K_b: Action.PRIORITISE_BEDFLOW,
    pygame.K_PERIOD: Action.WAIT,
}
# 'S' is walk-down; discharge support gets its own key.
KEY_ACTIONS[pygame.K_p] = Action.SUPPORT_DISCHARGE
# Right-arrow separately, since 'd' is taken by the dashboard.
KEY_ACTIONS[pygame.K_RIGHT] = Action.MOVE_RIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument(
        "--watch", action="store_true", help="let the rule-based policy play it"
    )
    args = parser.parse_args()

    cfg = SimConfig()
    cfg.telemetry_enabled = not args.no_telemetry
    engine = WardEngine(cfg, seed=args.seed)
    renderer = WardRenderer(engine)
    agent = RuleBasedAgent()
    agent.reset()

    clock = pygame.time.Clock()
    finished = False

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    renderer.close()
                    return
                if event.key == pygame.K_TAB:
                    action = agent.act(engine)
                elif event.key in KEY_ACTIONS:
                    action = KEY_ACTIONS[event.key]

        if args.watch and not finished:
            action = agent.act(engine)

        if action is not None and not finished:
            _obs, _reward, terminated, truncated, info = engine.step(int(action))
            if terminated or truncated:
                finished = True
                kpi = info["kpi"]
                print(f"\nShift over: {info['termination_reason']}")
                print(f"  return                 {info['total_reward']:+.1f}")
                print(f"  incident-free shift    {kpi['incident_free_shift']}")
                print(f"  time below range       {kpi['time_below_range_steps']} patient-steps")
                print(f"  alarms raised          {kpi['alarms_raised']}")
                print(f"  discharges             {kpi['discharges']}")

        renderer.draw()
        renderer.flip()
        clock.tick(30 if args.watch else 60)


if __name__ == "__main__":
    main()
