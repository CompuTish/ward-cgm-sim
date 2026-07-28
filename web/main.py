"""Browser entrypoint for the ward CGM telemetry simulator (pygbag/WebAssembly).

pygbag requires an async main loop that yields to the browser every frame, and
it packages only what lives under this directory - which is why
``scripts/build_web.py`` vendors the simulation package in here before building.

Nothing in this file may import numpy, gymnasium or any other native
dependency; see tests/test_web_bundle.py, which enforces that.

ACADEMIC MODEL - not clinical decision support.
"""

import asyncio

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
        self.renderer = WardRenderer(self.engine)
        self.agent = RuleBasedAgent()
        self.agent.reset()
        self.watching = True  # start by demonstrating the rule-based nurse
        self.finished = False
        self.frame = 0
        self.seed = 1

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
            "TAB take over / hand back    F5 new shift    F6 toggle telemetry",
            "arrows/WASD move    M dashboard    C check patient    N notes",
            "K consent    E enrol    R review eligibility    X de-enrol",
            "SPACE alarm    G point-of-care    1 treat hypo    2 treat hyper",
            "Q escalate    T troubleshoot sensor    P support discharge",
            "B prioritise bed flow    . wait    F1-F4 ask HCA/nurse/doctor/surgeon",
        ]
        if self.finished:
            hints.insert(0, "SHIFT COMPLETE - press any key for a new shift")
        return hints

    def draw_overlay(self) -> None:
        surface = self.renderer.surface
        font = self.renderer.font_small
        hints = self.hint_lines()
        banner = self.finished

        # Solid backing panel: these hints sit over the pale ward floor, and
        # without it the text is unreadable.
        line_height = font.get_linesize()
        height = line_height * len(hints) + 14
        width = max(font.size(line)[0] for line in hints) + 28
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((18, 21, 28, 232))
        pygame.draw.rect(panel, (70, 80, 98), panel.get_rect(), width=1)
        surface.blit(panel, (8, surface.get_height() - height - 8))

        y = surface.get_height() - height - 3
        for index, line in enumerate(hints):
            colour = (255, 214, 110) if (banner and index == 0) else (198, 206, 220)
            surface.blit(font.render(line, True, colour), (22, y))
            y += line_height


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
