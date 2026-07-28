"""Coverage for the browser entrypoint - the file that had none.

`web/main.py` owns the keyboard map and the watch/play state machine, and it
was completely untested. That is how D came to be bound to CHECK_DASHBOARD
while the on-screen help promised WASD movement: nothing anywhere asserted that
the keys did what the help said, so the nurse simply could not walk right.

These tests import the module rather than run it, which is why `main.py` guards
its `asyncio.run` behind `if __name__ == "__main__"`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from ward_cgm_sim.core.actions import Action  # noqa: E402

REPO_ROOT = Path(__file__).parents[1]
WEB_MAIN = REPO_ROOT / "web" / "main.py"


@pytest.fixture(scope="module")
def web_main():
    """Import web/main.py without letting its frame loop take over."""
    pygame.init()
    pygame.display.set_mode((8, 8))
    spec = importlib.util.spec_from_file_location("ward_web_main", WEB_MAIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ward_web_main"] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules["ward_web_main"]
    pygame.display.quit()


def key_event(key: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=key, unicode="", mod=0)


# --------------------------------------------------------------------------
# The keyboard map
# --------------------------------------------------------------------------


def test_wasd_covers_all_four_directions(web_main):
    """The regression that started this file.

    D was bound to CHECK_DASHBOARD, so 'WASD move' was a lie and the player
    could walk up, down and left but never right.
    """
    keys = web_main.KEY_ACTIONS
    assert keys[pygame.K_w] is Action.MOVE_UP
    assert keys[pygame.K_a] is Action.MOVE_LEFT
    assert keys[pygame.K_s] is Action.MOVE_DOWN
    assert keys[pygame.K_d] is Action.MOVE_RIGHT

    wasd = {keys[k] for k in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d)}
    assert wasd == {Action.MOVE_UP, Action.MOVE_LEFT,
                    Action.MOVE_DOWN, Action.MOVE_RIGHT}


def test_the_arrow_keys_cover_all_four_directions(web_main):
    keys = web_main.KEY_ACTIONS
    arrows = {keys[k] for k in (pygame.K_UP, pygame.K_DOWN,
                                pygame.K_LEFT, pygame.K_RIGHT)}
    assert arrows == {Action.MOVE_UP, Action.MOVE_DOWN,
                      Action.MOVE_LEFT, Action.MOVE_RIGHT}


def test_every_action_in_the_space_is_reachable_from_the_keyboard(web_main):
    """A player must be able to do everything the policy can do."""
    bound = set(web_main.KEY_ACTIONS.values())
    missing = set(Action) - bound
    assert not missing, f"no key reaches: {sorted(a.name for a in missing)}"


def test_no_key_is_bound_to_two_different_actions(web_main):
    """A dict cannot hold a duplicate key, so check the source instead."""
    source = WEB_MAIN.read_text(encoding="utf-8")
    body = source.split("KEY_ACTIONS = {", 1)[1].split("}", 1)[0]
    names = [line.split(":")[0].strip() for line in body.splitlines()
             if ":" in line and not line.strip().startswith("#")]
    assert len(names) > 20, "positive control: the map must have been parsed"
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"bound twice: {duplicates}"


def test_the_reserved_control_keys_are_not_also_actions(web_main):
    """TAB, F5 and F6 are handled before the action map and must stay free."""
    for key in (pygame.K_TAB, pygame.K_F5, pygame.K_F6):
        assert key not in web_main.KEY_ACTIONS


def test_the_on_screen_help_names_the_keys_that_are_actually_bound(web_main):
    """The help text is the only instruction a player gets.

    D was documented as the dashboard long after it stopped being bound to it;
    tie the two together so they cannot drift apart again.
    """
    demo = web_main.Demo()
    hints = " ".join(demo.hint_lines())
    keys = web_main.KEY_ACTIONS

    documented = {
        "M": Action.CHECK_DASHBOARD,
        "C": Action.CHECK_PATIENT,
        "N": Action.REVIEW_NOTES,
        "K": Action.ASK_CONSENT,
        "E": Action.ENROL,
        "R": Action.REVIEW_ELIGIBILITY,
        "X": Action.DEENROL,
        "G": Action.POC_GLUCOSE_TEST,
        "Q": Action.ESCALATE,
        "T": Action.TROUBLESHOOT_SENSOR,
        "P": Action.SUPPORT_DISCHARGE,
        "B": Action.PRIORITISE_BEDFLOW,
    }
    assert len(documented) == 12, "positive control"
    for letter, action in documented.items():
        assert keys[getattr(pygame, f"K_{letter.lower()}")] is action, letter
        assert f"{letter} " in hints, f"{letter} is bound but never explained"

    assert "WASD" in hints
    assert "D dashboard" not in hints, "the help still claims the old binding"


# --------------------------------------------------------------------------
# The watch / play state machine
# --------------------------------------------------------------------------


def test_the_demo_starts_by_watching_the_rule_based_nurse(web_main):
    demo = web_main.Demo()
    assert demo.watching is True
    assert demo.finished is False


def test_tab_hands_control_over_and_back(web_main):
    demo = web_main.Demo()
    demo.handle(key_event(pygame.K_TAB))
    assert demo.watching is False
    demo.handle(key_event(pygame.K_TAB))
    assert demo.watching is True


def test_pressing_an_action_key_takes_control_and_advances_one_step(web_main):
    demo = web_main.Demo()
    assert demo.watching is True
    before = demo.engine.step_index
    demo.handle(key_event(pygame.K_PERIOD))  # WAIT
    assert demo.watching is False, "acting should take over from the policy"
    assert demo.engine.step_index == before + 1


def test_an_unbound_key_does_nothing_at_all(web_main):
    demo = web_main.Demo()
    before = demo.engine.step_index
    demo.handle(key_event(pygame.K_z))
    assert demo.engine.step_index == before
    assert demo.watching is True


def test_only_keydown_events_are_acted_on(web_main):
    demo = web_main.Demo()
    before = demo.engine.step_index
    demo.handle(pygame.event.Event(pygame.KEYUP, key=pygame.K_PERIOD))
    demo.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(0, 0), button=1))
    assert demo.engine.step_index == before


def test_walking_right_actually_moves_the_nurse_right(web_main):
    """End to end for the reported bug: press D, the nurse moves right."""
    demo = web_main.Demo()
    engine = demo.engine
    # Somewhere with clear floor to the right.
    start = next(
        (x, y)
        for y in range(engine.ward_map.height)
        for x in range(engine.ward_map.width)
        if engine.ward_map.walkable(x, y) and engine.ward_map.walkable(x + 1, y)
    )
    engine.agent_x, engine.agent_y = start
    demo.handle(key_event(pygame.K_d))
    assert (engine.agent_x, engine.agent_y) == (start[0] + 1, start[1])


def test_every_movement_key_moves_the_nurse_the_right_way(web_main):
    deltas = {
        pygame.K_w: (0, -1), pygame.K_UP: (0, -1),
        pygame.K_s: (0, 1), pygame.K_DOWN: (0, 1),
        pygame.K_a: (-1, 0), pygame.K_LEFT: (-1, 0),
        pygame.K_d: (1, 0), pygame.K_RIGHT: (1, 0),
    }
    for key, (dx, dy) in deltas.items():
        demo = web_main.Demo()
        engine = demo.engine
        start = next(
            (x, y)
            for y in range(engine.ward_map.height)
            for x in range(engine.ward_map.width)
            if engine.ward_map.walkable(x, y) and engine.ward_map.walkable(x + dx, y + dy)
        )
        engine.agent_x, engine.agent_y = start
        demo.handle(key_event(key))
        assert (engine.agent_x, engine.agent_y) == (start[0] + dx, start[1] + dy), (
            f"{pygame.key.name(key)} did not move ({dx}, {dy})"
        )


def test_f5_starts_a_new_shift_on_a_new_seed(web_main):
    demo = web_main.Demo()
    demo.handle(key_event(pygame.K_PERIOD))
    assert demo.engine.step_index == 1
    seed = demo.seed
    demo.handle(key_event(pygame.K_F5))
    assert demo.seed != seed
    assert demo.engine.step_index == 0


def test_f6_toggles_telemetry_and_restarts(web_main):
    demo = web_main.Demo()
    before = demo.config.telemetry_enabled
    demo.handle(key_event(pygame.K_F6))
    assert demo.config.telemetry_enabled is not before
    assert demo.engine.cfg.telemetry_enabled is not before
    assert demo.engine.step_index == 0
    demo.handle(key_event(pygame.K_F6))
    assert demo.config.telemetry_enabled is before


def test_a_finished_shift_restarts_on_any_key(web_main):
    demo = web_main.Demo()
    demo.finished = True
    demo.handle(key_event(pygame.K_PERIOD))
    assert demo.finished is False
    assert demo.engine.step_index == 0


def test_advance_does_nothing_once_the_shift_is_over(web_main):
    demo = web_main.Demo()
    demo.finished = True
    before = demo.engine.step_index
    demo.advance(Action.WAIT)
    assert demo.engine.step_index == before


def test_a_full_shift_can_be_played_to_completion(web_main):
    """End to end: drive the demo with its own policy until the shift ends."""
    demo = web_main.Demo()
    for _ in range(200):
        if demo.finished:
            break
        demo.advance(demo.agent.act(demo.engine))
    assert demo.finished, "the shift never ended"
    assert demo.engine.step_index <= demo.config.steps_per_episode


def test_drawing_a_frame_with_the_overlay_does_not_raise(web_main):
    demo = web_main.Demo()
    demo.renderer.draw()
    demo.draw_overlay()
    surface = demo.renderer.surface
    seen = {
        surface.get_at((x, y))[:3]
        for x in range(0, surface.get_width(), 7)
        for y in range(0, surface.get_height(), 7)
    }
    assert len(seen) > 20, "the frame came out nearly blank"


def test_the_help_panel_fits_inside_the_canvas(web_main):
    """Hints are laid out from measured metrics; a change must not push them off."""
    demo = web_main.Demo()
    surface = demo.renderer.surface
    font = demo.renderer.font_small
    lines = demo.hint_lines()
    assert lines, "positive control: there must be hints to lay out"
    widest = max(font.size(line)[0] for line in lines)
    height = font.get_linesize() * len(lines) + 14
    assert widest + 28 <= surface.get_width(), "the help panel is wider than the canvas"
    assert height <= surface.get_height(), "the help panel is taller than the canvas"
