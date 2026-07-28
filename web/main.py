"""Browser entrypoint for the ward CGM telemetry simulator (pygbag/WebAssembly).

pygbag requires an async main loop that yields to the browser every frame, and
it packages only what lives under this directory - which is why
``scripts/build_web.py`` vendors the simulation package in here before building.

Nothing in this file may import numpy, gymnasium or any other native
dependency; see tests/test_web_bundle.py, which enforces that.

ACADEMIC MODEL - not clinical decision support.
"""

import asyncio
import json
import sys

import pygame

from ward_cgm_sim.agents.rule_based import RuleBasedAgent
from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.actions import Action
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.render.pygame_renderer import WardRenderer

# Steps per second while the rule-based policy is driving. Deliberately slow:
# at six steps a second the ward changes faster than a viewer can follow what
# the nurse is actually doing.
WATCH_SPEED = 3

# The page that is allowed to receive the readout. Addressed explicitly rather
# than "*" so the state is never posted to whatever happens to be framing the
# demo. The demo lives on its own origin deliberately - see the deploy notes -
# and this one-way channel is the only thing that crosses it.
PARENT_ORIGIN = "https://isabelsmith.me"

# Frames between readout updates. The simulation advances at most a few times a
# second, so posting every frame would be 30x the traffic for no new numbers.
PUBLISH_EVERY = 10

# The hosting page appends this to the iframe URL to say "I am rendering the
# readout myself". A fragment rather than frame detection because it is
# deterministic and fails safe: anyone opening the demo directly has no
# fragment, so they keep the in-canvas HUD and lose nothing.
EXTERNAL_PANEL_FRAGMENT = "panel=external"


def external_panel_requested() -> bool:
    """True when the page around us is rendering the readout instead.

    Read by several routes because this is the one thing here that cannot be
    exercised off the browser: if none of them works the answer is False, the
    canvas keeps its own HUD, and the demo is merely letterboxed rather than
    left with no readout at all.
    """
    if sys.platform != "emscripten":
        return False
    try:
        import platform as _platform

        location = _platform.window.location
        for attribute in ("hash", "href", "search"):
            try:
                value = str(getattr(location, attribute, "") or "")
            except Exception:
                continue
            if EXTERNAL_PANEL_FRAGMENT in value:
                return True
    except Exception:  # pragma: no cover - browser-only path
        return False
    return False

KEY_ACTIONS = {
    pygame.K_UP: Action.MOVE_UP,
    pygame.K_w: Action.MOVE_UP,
    pygame.K_DOWN: Action.MOVE_DOWN,
    pygame.K_s: Action.MOVE_DOWN,
    pygame.K_LEFT: Action.MOVE_LEFT,
    pygame.K_a: Action.MOVE_LEFT,
    pygame.K_RIGHT: Action.MOVE_RIGHT,
    # D completes WASD. The dashboard moved to M (monitor): bound to D it left
    # the nurse unable to walk right while the on-screen help promised WASD.
    pygame.K_d: Action.MOVE_RIGHT,
    pygame.K_m: Action.CHECK_DASHBOARD,
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
    pygame.K_p: Action.SUPPORT_DISCHARGE,
    pygame.K_b: Action.PRIORITISE_BEDFLOW,
    pygame.K_PERIOD: Action.WAIT,
}


class Demo:
    def __init__(self):
        self.config = SimConfig()
        self.engine = WardEngine(self.config, seed=1)
        self.external_panel = external_panel_requested()
        self.renderer = WardRenderer(self.engine, show_hud=not self.external_panel)
        self.agent = RuleBasedAgent()
        self.agent.reset()
        self.watching = True  # start by demonstrating the rule-based nurse
        self.finished = False
        self.frame = 0
        self.seed = 1
        # Whether the hosting page has ever received the readout, and why not.
        # While this is False the canvas keeps drawing its own compact readout,
        # so a broken channel never leaves a ward with no numbers at all.
        self.published_ok = False
        self.publish_error = ""
        self._ack_installed = False

    def restart(self, telemetry: bool | None = None):
        if telemetry is not None:
            self.config.telemetry_enabled = telemetry
        self.seed += 1
        self.engine.cfg = self.config
        self.engine.reset(self.seed)
        self.renderer.engine = self.engine
        self.agent.reset()
        self.finished = False

    def advance(self, action):
        if self.finished:
            return
        _obs, _reward, terminated, truncated, _info = self.engine.step(int(action))
        if terminated or truncated:
            self.finished = True

    def handle(self, event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_TAB:
            self.watching = not self.watching
            return
        if event.key == pygame.K_F5:
            self.restart()
            return
        if event.key == pygame.K_F6:
            self.restart(telemetry=not self.config.telemetry_enabled)
            return
        if self.finished:
            self.restart()
            return
        if event.key in KEY_ACTIONS:
            self.watching = False
            self.advance(KEY_ACTIONS[event.key])

    def state(self) -> dict:
        """The readout, as data, for the page hosting the demo.

        This mirrors exactly what `_draw_hud` already paints inside the canvas
        and adds nothing: the same counts, the same coarse staff summary, and
        only the alarms `visible_alarms()` returns. Publishing anything the HUD
        does not show would hand the viewer a fact the policy has to spend a
        step learning, which is the same POMDP boundary the renderer observes.
        """
        engine = self.engine
        cfg = engine.cfg
        flow = engine.flow
        minutes = engine.step_index * cfg.minutes_per_step
        staff_words = ("skeleton", "stretched", "comfortable")

        # `visible_alarms()` is the single source of truth for what the agent
        # can see - it already returns nothing when telemetry is off, and
        # nothing off-station that has not been looked at. Re-checking any of
        # that here would just be a second copy of the rule, free to drift.
        alarms = [
            {
                "bed": alarm.bed,
                "kind": alarm.kind.value,
                "value": round(alarm.cgm_value, 1),
                "age": alarm.age(engine.step_index),
                "urgent": bool(alarm.is_urgent),
            }
            for alarm in engine.visible_alarms()[:8]
        ]

        return {
            "type": "ward-cgm-sim",
            "version": 1,
            "clock": f"{7 + minutes // 60:02d}:{minutes % 60:02d}",
            "step": engine.step_index,
            "steps": cfg.steps_per_episode,
            "beds": flow.occupied_beds,
            "capacity": flow.n_beds,
            "free": flow.free_beds,
            "queue": flow.queue_length,
            "enrolled": sum(1 for p in flow.patients() if p.is_enrolled),
            "staff": staff_words[engine.staff.coarse_availability()],
            "telemetry": bool(cfg.telemetry_enabled),
            "boardRead": bool(engine.ward_map.at_station(engine.agent_x, engine.agent_y)),
            "alarms": alarms,
            "watching": bool(self.watching),
            "finished": bool(self.finished),
            "lastAction": engine.last_action_result,
            "return": round(engine.rewards.total, 1),
        }

    def publish(self) -> bool:
        """Send the readout to the hosting page. False when it did not arrive.

        The demo is served from its own origin on purpose, so the page cannot
        reach into this frame - a one-way postMessage is the whole channel, and
        it is addressed to a single origin rather than "*".

        Sent is not the same as delivered. postMessage does not raise when the
        target turns out not to be at the origin we addressed, so a send can
        succeed while nothing ever arrives - which is exactly what happened.
        The page therefore acknowledges, and only that acknowledgement counts.
        Both `parent` and `top` are posted to, because how deeply the runtime
        frames the canvas is not something we can establish from here; the
        pinned origin means only the real page can receive either one.
        """
        if sys.platform != "emscripten":
            self.publish_error = "not running in a browser"
            return False

        try:
            import platform as _platform

            window = _platform.window
        except Exception as error:  # pragma: no cover - browser-only path
            self.publish_error = "no window: %s" % error
            return False

        if not self._ack_installed:
            try:  # a listener that records the page's acknowledgement
                window.eval(
                    "window.__wardAck=0;window.addEventListener('message',"
                    "function(e){if(e.origin===%r&&e.data==='ward-cgm-sim-ack')"
                    "window.__wardAck=1});" % PARENT_ORIGIN
                )
                self._ack_installed = True
            except Exception as error:  # pragma: no cover - browser-only path
                self.publish_error = "no ack listener: %s" % error

        payload = json.dumps(self.state())
        errors = []
        sent = False
        for name in ("parent", "top"):
            try:
                target = getattr(window, name, None)
                if target is None:
                    errors.append("%s missing" % name)
                    continue
                target.postMessage(payload, PARENT_ORIGIN)
                sent = True
            except Exception as error:  # pragma: no cover - browser-only path
                errors.append("%s: %s" % (name, error))

        if not sent:
            try:
                helper = "window.__wardPost=function(s){window.parent.postMessage(s,%r);window.top.postMessage(s,%r)}" % (PARENT_ORIGIN, PARENT_ORIGIN)
                window.eval(helper)
                window.__wardPost(payload)
                sent = True
            except Exception as error:  # pragma: no cover - browser-only path
                errors.append("eval: %s" % error)

        try:
            acknowledged = bool(getattr(window, "__wardAck", 0))
        except Exception:  # pragma: no cover - browser-only path
            acknowledged = False

        if acknowledged:
            self.published_ok = True
            self.publish_error = ""
            return True

        self.published_ok = False
        self.publish_error = (
            "sent, no reply from the page yet" if sent
            else ("; ".join(errors)[:110] or "nowhere to send to")
        )
        return False

    def hint_lines(self) -> list[str]:
        """The on-screen help, as text.

        Separate from drawing it so the tests can check that what the player is
        told matches what KEY_ACTIONS actually does - the two drifted apart
        once already, leaving D documented as the dashboard while it moved to
        movement. See tests/test_web_main.py.
        """
        mode = "watching the rule-based nurse" if self.watching else "you are the nurse"
        telemetry = "telemetry ON" if self.config.telemetry_enabled else "telemetry OFF"
        hints = [
            f"{mode}  -  {telemetry}",
            "TAB take control or hand back to the nurse    F5 new shift",
            "F6 toggle telemetry    arrows/WASD move    M dashboard",
            "C check patient    N notes",
            "K consent    E enrol    R review eligibility    X de-enrol",
            "SPACE alarm    G point-of-care    1 treat hypo    2 treat hyper",
            "Q escalate    T troubleshoot sensor    P support discharge",
            "B prioritise bed flow    . wait    F1-F4 ask HCA/nurse/doctor/surgeon",
        ]
        if self.finished:
            hints.insert(0, "SHIFT COMPLETE - press any key for a new shift")
        return hints

    def draw_overlay(self) -> None:
        """Whatever the canvas still has to say for itself.

        When the hosting page is showing the readout and receiving it, the
        canvas says nothing: the controls and the numbers are all beside it in
        real text. It speaks up only when there is no page listening, or when
        the readout is not getting through - a silent channel must not leave a
        ward with no numbers on it.
        """
        if self.external_panel and self.published_ok and not self.finished:
            return

        surface = self.renderer.surface
        font = self.renderer.font_small

        if self.external_panel:
            lines = list(self.fallback_lines())
        else:
            lines = self.hint_lines()

        line_height = font.get_linesize()
        height = line_height * len(lines) + 14
        width = max(font.size(line)[0] for line in lines) + 28
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((18, 21, 28, 232))
        pygame.draw.rect(panel, (70, 80, 98), panel.get_rect(), width=1)
        surface.blit(panel, (8, surface.get_height() - height - 8))

        y = surface.get_height() - height - 3
        for index, line in enumerate(lines):
            highlight = self.finished and index == 0
            colour = (255, 214, 110) if highlight else (198, 206, 220)
            surface.blit(font.render(line, True, colour), (22, y))
            y += line_height

    def fallback_lines(self) -> list[str]:
        """A compact readout for when the page is not receiving one."""
        state = self.state()
        lines = []
        if self.finished:
            lines.append("SHIFT COMPLETE - press any key for a new shift")
        lines += [
            "%s  step %d/%d  -  beds %d/%d  free %d  queue %d  enrolled %d"
            % (state["clock"], state["step"], state["steps"], state["beds"],
               state["capacity"], state["free"], state["queue"], state["enrolled"]),
            "staff %s  -  return %+.1f  -  %s"
            % (state["staff"], state["return"], state["lastAction"]),
        ]
        if not self.published_ok:
            lines.append("the panel beside this ward is not receiving: %s"
                         % (self.publish_error or "no reply yet"))
        lines.append("academic model - not clinical advice")
        return lines


async def main() -> None:
    pygame.init()
    demo = Demo()
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            demo.handle(event)

        demo.frame += 1
        if demo.watching and not demo.finished:
            if demo.frame % max(1, 30 // WATCH_SPEED) == 0:
                demo.advance(demo.agent.act(demo.engine))

        demo.renderer.draw()
        demo.draw_overlay()
        demo.renderer.flip()

        if demo.frame % PUBLISH_EVERY == 0:
            demo.publish()

        clock.tick(30)
        # Yielding here is what lets the browser stay responsive; pygbag
        # requires it once per frame.
        await asyncio.sleep(0)


# pygbag runs this file as the entry script, so this still starts the demo in
# the browser - but it also lets the tests import Demo and KEY_ACTIONS without
# the frame loop taking over the process. Without the guard this module could
# not be imported at all, which is why none of it had any coverage.
if __name__ == "__main__":
    asyncio.run(main())
