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


# --------------------------------------------------------------------------
# The live readout published to the hosting page
# --------------------------------------------------------------------------


def test_the_published_state_is_json_serialisable(web_main):
    """It crosses the origin boundary as a JSON string, so it has to encode."""
    import json

    demo = web_main.Demo()
    encoded = json.dumps(demo.state())
    round_tripped = json.loads(encoded)
    assert round_tripped["type"] == "ward-cgm-sim"
    assert round_tripped["version"] == 1


def test_the_published_state_matches_what_the_canvas_hud_shows(web_main):
    demo = web_main.Demo()
    for _ in range(25):
        demo.advance(demo.agent.act(demo.engine))
    state = demo.state()
    engine, flow = demo.engine, demo.engine.flow

    assert state["step"] == engine.step_index
    assert state["steps"] == demo.config.steps_per_episode
    assert state["beds"] == flow.occupied_beds
    assert state["capacity"] == flow.n_beds
    assert state["free"] == flow.free_beds
    assert state["queue"] == flow.queue_length
    assert state["enrolled"] == sum(1 for p in flow.patients() if p.is_enrolled)
    assert state["staff"] in ("skeleton", "stretched", "comfortable")
    assert state["lastAction"] == engine.last_action_result
    assert state["return"] == round(engine.rewards.total, 1)


def test_the_clock_matches_the_step_it_was_taken_at(web_main):
    demo = web_main.Demo()
    assert demo.state()["clock"] == "07:00"
    for _ in range(12):  # twelve five-minute steps = one hour
        demo.advance(web_main.Action.WAIT)
    assert demo.state()["clock"] == "08:00"


def test_the_readout_only_carries_alarms_the_agent_can_already_see(web_main):
    """The same POMDP boundary the renderer keeps.

    An alarm that has not been seen on the board is not knowledge, and
    publishing it to the page would hand the viewer a fact the policy has to
    spend a step acquiring.
    """
    demo = web_main.Demo()
    engine = demo.engine
    for _ in range(demo.config.steps_per_episode - 1):
        demo.advance(web_main.Action.CHECK_DASHBOARD)
        visible = engine.visible_alarms()
        published = demo.state()["alarms"]
        assert len(published) == min(8, len(visible))
        assert {a["bed"] for a in published} <= {a.bed for a in visible}
        if engine.active_alarms and visible:
            # Positive control: alarms did exist and were published.
            assert published
            return
    raise AssertionError("no alarm arose in a whole shift; the check was vacuous")


def test_no_hidden_clinical_fact_reaches_the_page(web_main):
    """Whatever else changes, the payload must stay a summary."""
    demo = web_main.Demo()
    for _ in range(40):
        demo.advance(demo.agent.act(demo.engine))
    allowed = {
        "type", "version", "clock", "step", "steps", "beds", "capacity", "free",
        "queue", "enrolled", "staff", "telemetry", "boardRead", "alarms",
        "watching", "finished", "lastAction", "return",
    }
    state = demo.state()
    assert set(state) == allowed, f"unexpected keys: {set(state) ^ allowed}"

    banned = ("true_glucose", "diabetes", "capacity_to_consent", "insulin",
              "discharge_stage", "expected_los", "end_of_life", "pregnan")
    import json

    blob = json.dumps(state).lower()
    for term in banned:
        assert term not in blob, f"the readout leaks {term}"

    for alarm in state["alarms"]:
        assert set(alarm) == {"bed", "kind", "value", "age", "urgent"}


def test_the_readout_follows_visible_alarms_exactly(web_main):
    """One source of truth for what the agent can see.

    The readout deliberately does not re-implement any visibility rule; it
    republishes `visible_alarms()`. Injecting an alarm the board has not shown
    proves that, where simply running a shift would not: with telemetry off no
    alarm is ever raised, so an assertion that none is published passes even if
    the rule is gone.
    """
    from ward_cgm_sim.core.alarms import Alarm, AlarmKind

    demo = web_main.Demo()
    engine = demo.engine
    engine.active_alarms[0] = Alarm(
        bed=0, kind=AlarmKind.HYPO, raised_step=engine.step_index, cgm_value=3.4
    )
    # Standing away from the station, with the board never checked, it is
    # invisible - and so must not be published.
    engine.dashboard_seen_step = None
    engine.agent_x, engine.agent_y = 1, 1
    assert not engine.ward_map.at_station(1, 1), "positive control: away from the board"
    assert engine.visible_alarms() == []
    assert demo.state()["alarms"] == []

    # Read the board and the very same alarm becomes publishable.
    engine.step(int(Action.CHECK_DASHBOARD))
    assert engine.visible_alarms(), "positive control: the alarm is now on the board"
    published = demo.state()["alarms"]
    assert [a["bed"] for a in published] == [a.bed for a in engine.visible_alarms()]


def test_telemetry_off_publishes_no_alarms_at_all(web_main):
    demo = web_main.Demo()
    demo.handle(key_event(pygame.K_F6))
    assert demo.config.telemetry_enabled is False
    for _ in range(60):
        demo.advance(demo.agent.act(demo.engine))
        state = demo.state()
        assert state["telemetry"] is False
        assert state["alarms"] == []


def test_publishing_is_a_no_op_off_the_browser(web_main):
    """Nothing to post to on a desktop, and it must not raise there."""
    demo = web_main.Demo()
    assert demo.publish() is False


def test_the_readout_is_addressed_to_one_origin_not_a_wildcard(web_main):
    """A wildcard target would post the state to whatever framed the demo."""
    assert web_main.PARENT_ORIGIN == "https://isabelsmith.me"
    assert web_main.PARENT_ORIGIN.startswith("https://")

    source = WEB_MAIN.read_text(encoding="utf-8")
    calls = [
        line.strip() for line in source.splitlines()
        if "postMessage(" in line and not line.strip().startswith("#")
    ]
    assert calls, "positive control: there must be a postMessage call to check"
    for call in calls:
        assert "'*'" not in call and '"*"' not in call, f"wildcard target: {call}"
        assert "PARENT_ORIGIN" in call or "%s" in call, f"untargeted postMessage: {call}"


def test_the_readout_is_not_published_every_single_frame(web_main):
    """30 posts a second for numbers that change a few times a second."""
    assert web_main.PUBLISH_EVERY >= 5


def test_the_external_panel_switch_defaults_to_keeping_the_canvas_hud(web_main):
    """Fail-safe: off the browser, and on any error, the HUD stays.

    The page hides the canvas HUD by asking for it in the URL. If that request
    can never be read the demo must still show its own readout, or a viewer
    would be left with a ward and no numbers at all.
    """
    assert web_main.external_panel_requested() is False
    demo = web_main.Demo()
    assert demo.external_panel is False
    assert demo.renderer.show_hud is True


def test_the_fragment_the_page_appends_is_the_one_the_demo_looks_for(web_main):
    """Two files have to agree on this string or the switch silently does nothing."""
    page = (
        REPO_ROOT.parent / "site_isabelsmith.me" / "public" / "projects"
        / "ward-sim" / "index.html"
    )
    if not page.is_file():
        pytest.skip("the project page lives in the parent repository")
    markup = page.read_text(encoding="utf-8")
    assert "ward-cgm-demo.web.app/#" + web_main.EXTERNAL_PANEL_FRAGMENT in markup, (
        "the page does not ask for the external panel with the expected fragment"
    )


# --------------------------------------------------------------------------
# What the canvas still draws once the page owns the readout
# --------------------------------------------------------------------------


def test_the_canvas_stays_quiet_once_the_page_is_receiving(web_main):
    """No help box over the ward when the page shows the controls in a tab."""
    demo = web_main.Demo()
    demo.external_panel = True
    demo.published_ok = True
    before = pygame.image.tobytes(demo.renderer.surface, "RGB")
    demo.draw_overlay()
    assert pygame.image.tobytes(demo.renderer.surface, "RGB") == before, (
        "the canvas drew an overlay it no longer needs"
    )


def test_the_canvas_keeps_its_own_help_when_no_page_is_listening(web_main):
    """Opened directly, the demo is the whole interface and must explain itself."""
    demo = web_main.Demo()
    assert demo.external_panel is False
    before = pygame.image.tobytes(demo.renderer.surface, "RGB")
    demo.draw_overlay()
    assert pygame.image.tobytes(demo.renderer.surface, "RGB") != before
    assert any("WASD" in line for line in demo.hint_lines())


def test_a_silent_channel_still_gets_numbers_onto_the_ward(web_main):
    """The regression this exists for.

    With the page rendering the readout, the canvas HUD is gone. If the
    readout never arrives the viewer is left with a ward and no numbers, which
    is exactly what happened. While nothing has got through, the canvas draws a
    compact readout and says why.
    """
    demo = web_main.Demo()
    demo.external_panel = True
    demo.published_ok = False
    demo.publish_error = "parent missing"
    for _ in range(20):
        demo.advance(demo.agent.act(demo.engine))

    lines = demo.fallback_lines()
    joined = " ".join(lines)
    assert "beds" in joined and "queue" in joined and "enrolled" in joined
    assert "step %d" % demo.engine.step_index in joined
    assert "not receiving" in joined and "parent missing" in joined
    assert "not clinical advice" in joined

    before = pygame.image.tobytes(demo.renderer.surface, "RGB")
    demo.draw_overlay()
    assert pygame.image.tobytes(demo.renderer.surface, "RGB") != before, (
        "nothing was drawn, so the ward has no numbers on it at all"
    )


def test_the_fallback_stops_apologising_once_the_channel_works(web_main):
    demo = web_main.Demo()
    demo.external_panel = True
    demo.published_ok = True
    assert not any("not receiving" in line for line in demo.fallback_lines())


def test_the_end_of_shift_banner_shows_even_when_the_page_owns_the_readout(web_main):
    """Otherwise a finished shift looks like a frozen one."""
    demo = web_main.Demo()
    demo.external_panel = True
    demo.published_ok = True
    demo.finished = True
    assert demo.fallback_lines()[0].startswith("SHIFT COMPLETE")
    before = pygame.image.tobytes(demo.renderer.surface, "RGB")
    demo.draw_overlay()
    assert pygame.image.tobytes(demo.renderer.surface, "RGB") != before


def test_publish_records_why_it_could_not_send(web_main):
    demo = web_main.Demo()
    assert demo.publish() is False
    assert demo.publish_error, "a failure with no reason cannot be diagnosed"
    assert demo.published_ok is False


def test_the_help_no_longer_says_hand_back_without_saying_to_whom(web_main):
    demo = web_main.Demo()
    hints = " ".join(demo.hint_lines())
    assert "hand back to the nurse" in hints
    assert "take over / hand back" not in hints, "the old ambiguous wording is back"


def test_publish_reports_every_route_it_tried(web_main, monkeypatch):
    """The reason is what turns a blank panel into a diagnosis.

    Off the browser publish() returns early, so that path alone never
    exercises the error handling. Stand in a window whose every route fails.
    """
    import types

    class Failing:
        @property
        def parent(self):
            raise RuntimeError("parent blocked")

        @property
        def top(self):
            raise RuntimeError("top blocked")

        def eval(self, _source):
            raise RuntimeError("eval blocked")

    monkeypatch.setattr(web_main.sys, "platform", "emscripten")
    monkeypatch.setitem(
        sys.modules, "platform", types.SimpleNamespace(window=Failing())
    )

    demo = web_main.Demo()
    assert demo.publish() is False
    assert demo.published_ok is False
    for expected in ("parent blocked", "top blocked", "eval blocked"):
        assert expected in demo.publish_error, demo.publish_error


def test_publish_succeeds_through_the_first_route_that_works(web_main, monkeypatch):
    import types

    sent = []

    class Parent:
        def postMessage(self, payload, origin):
            sent.append((payload, origin))

    monkeypatch.setattr(web_main.sys, "platform", "emscripten")
    monkeypatch.setitem(
        sys.modules, "platform", types.SimpleNamespace(window=types.SimpleNamespace(parent=Parent()))
    )

    demo = web_main.Demo()
    assert demo.publish() is True
    assert demo.published_ok is True
    assert demo.publish_error == ""
    assert len(sent) == 1
    payload, origin = sent[0]
    assert origin == web_main.PARENT_ORIGIN
    import json as _json
    assert _json.loads(payload)["type"] == "ward-cgm-sim"
