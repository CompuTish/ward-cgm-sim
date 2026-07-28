"""Top-down ward renderer in the style of a 16-bit RPG overworld.

Layout: the tiled ward map on the left, a HUD panel on the right carrying the
shift clock, occupancy, queue and the telemetry board. Alarms pulse over the
bed they belong to; enrolled patients carry a small green sensor pip; a patient
whose signal has dropped out shows a grey question mark, because that is
exactly the cue the agent is supposed to notice.

Stdlib + pygame only (ships to the browser build).
"""

import pygame

from ..core.alarms import AlarmKind
from ..core.patient import EnrolmentStatus, Location
from ..core.ward_map import BED, DRUG_ROOM, ENTRANCE, FLOOR, STATION, WALL
from . import sprites
from .sprites import TILE, SpriteSheet, u

HUD_WIDTH = 340
PANEL_BG = (28, 32, 42)
PANEL_ALT = (38, 44, 56)
TEXT = (232, 236, 244)
TEXT_DIM = (150, 158, 172)
ACCENT = (94, 186, 214)

ALARM_COLOURS = {
    AlarmKind.SEVERE_HYPO: (232, 58, 64),
    AlarmKind.HYPO: (232, 106, 70),
    AlarmKind.RAPID_FALL: (236, 152, 62),
    AlarmKind.HYPER: (226, 178, 60),
    AlarmKind.RAPID_RISE: (206, 190, 90),
}


class WardRenderer:
    """Draws the engine state. One instance per window."""

    def __init__(self, engine, headless: bool = False, scale: int = 1):
        self.engine = engine
        self.scale = scale
        self.headless = headless

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        map_w = engine.ward_map.width * TILE
        map_h = engine.ward_map.height * TILE
        self.width = map_w + HUD_WIDTH
        self.height = max(map_h, 560)

        if headless:
            self.surface = pygame.Surface((self.width, self.height))
        else:
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Ward CGM telemetry simulator - academic model")

        self.sprites = SpriteSheet()
        # Sized against the tile so the HUD stays legible if TILE changes.
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)
        self.font_small = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.font_big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 26, bold=True)
        # Every HUD row is laid out from the measured line height rather than
        # hard-coded offsets, so a font change cannot silently overlap boxes.
        self.line = self.font.get_linesize()
        self.line_small = self.font_small.get_linesize()
        self.frame = 0

    # ------------------------------------------------------------------
    def draw(self) -> None:
        self.frame += 1
        self.surface.fill(PANEL_BG)
        self._draw_map()
        self._draw_hud()

    def flip(self) -> None:
        if not self.headless:
            pygame.display.flip()

    def to_rgb_array(self):
        """Return the frame as a (H, W, 3) array. Requires numpy (native only)."""
        import numpy as np

        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self) -> None:
        if not self.headless:
            pygame.display.quit()

    # ------------------------------------------------------------------
    def _draw_map(self) -> None:
        engine = self.engine
        ward_map = engine.ward_map

        for y in range(ward_map.height):
            for x in range(ward_map.width):
                tile = ward_map.tile(x, y)
                pos = (x * TILE, y * TILE)
                if tile == WALL:
                    self.surface.blit(self.sprites.wall, pos)
                elif tile == STATION:
                    self.surface.blit(self.sprites.station, pos)
                elif tile == DRUG_ROOM:
                    self.surface.blit(self.sprites.drug_room, pos)
                elif tile == ENTRANCE:
                    self.surface.blit(self.sprites.entrance, pos)
                elif tile == BED:
                    self._draw_bed(x, y)
                else:
                    self.surface.blit(self.sprites.floor[(x + y) % 2], pos)

        self._draw_queue()
        self._draw_walking_patients()
        self._draw_background_staff()
        self._draw_agent()

    def _draw_bed(self, x: int, y: int) -> None:
        engine = self.engine
        bed = engine.ward_map.tile_beds[(x, y)]
        patient = engine.flow.patient_at_bed(bed)
        pos = (x * TILE, y * TILE)

        if patient is None or patient.location is not Location.BED:
            self.surface.blit(self.sprites.bed, pos)
        else:
            # Skin and blanket are both keyed off the patient id, so a given
            # patient keeps the same appearance for the whole shift.
            skins = self.sprites.patients
            blankets = skins[patient.patient_id % len(skins)]
            sprite = blankets[patient.patient_id % len(blankets)]
            self.surface.blit(sprite, pos)

        # Bed number, small and dim.
        label = self.font_small.render(str(bed), True, (120, 126, 140))
        self.surface.blit(label, (x * TILE + u(2), y * TILE + TILE - u(12)))

        if patient is None:
            return

        # Enrolment pip.
        if patient.enrolment is EnrolmentStatus.ENROLLED:
            colour = sprites.SIGNAL_LOST if patient.signal_lost else sprites.ENROLLED_MARK
            pygame.draw.circle(self.surface, colour, (x * TILE + TILE - u(5), y * TILE + u(5)), u(3))
            if patient.steps_since_valid_cgm > self.engine.cfg.alarms.signal_loss_grace_steps:
                mark = self.font_small.render("?", True, sprites.SIGNAL_LOST)
                self.surface.blit(mark, (x * TILE + TILE - u(10), y * TILE + u(7)))

        # Alarm overlay, pulsing so it reads at a glance.
        alarm = engine.active_alarms.get(bed)
        if alarm is not None and alarm.resolved_step is None and engine.cfg.telemetry_enabled:
            colour = ALARM_COLOURS.get(alarm.kind, sprites.ALARM_URGENT)
            pulse = 2 if (self.frame // 4) % 2 == 0 else 1
            pygame.draw.rect(
                self.surface,
                colour,
                pygame.Rect(x * TILE, y * TILE, TILE, TILE),
                width=pulse + u(1),
                border_radius=u(3),
            )

    def _draw_queue(self) -> None:
        """Patients waiting for a bed, queued outside the entrance."""
        engine = self.engine
        ex, ey = engine.ward_map.entrance_tile
        for i, _patient in enumerate(engine.flow.queue[:8]):
            px = ex * TILE - (i % 4) * (TILE - u(4)) - u(6)
            py = ey * TILE + (i // 4) * u(10)
            sprite = self.sprites.walking_patient[(self.frame // 6 + i) % 2]
            self.surface.blit(sprite, (px, py))

    def _draw_walking_patients(self) -> None:
        engine = self.engine
        for patient in engine.flow.patients():
            if patient.location is not Location.WALKING:
                continue
            # Walk along the corridor between the entrance and the bed.
            bx, by = engine.ward_map.bed_tile(patient.bed)
            ex, ey = engine.ward_map.entrance_tile
            total = max(1, patient.walk_total_steps)
            progress = 1.0 - (patient.walk_steps_left / total)
            if patient.walk_purpose == "discharge":
                progress = 1.0 - progress
            px = int((ex + (bx - ex) * progress) * TILE)
            py = int((ey + (by - ey) * progress) * TILE)
            sprite = self.sprites.walking_patient[(self.frame // 6) % 2]
            self.surface.blit(sprite, (px, py))

    def _draw_background_staff(self) -> None:
        """Colleagues drifting around the ward: available staff are visible."""
        engine = self.engine
        anchors = {
            "hca": (5, 8),
            "nurse": (18, 8),
            "doctor": (9, 16),
            "surgeon": (17, 16),
            "diabetes": (21, 11),
        }
        # Staff sprites are drawn from the same coarse availability the agent
        # can observe, not from per-role truth. Rendering exactly which roles
        # are free would show the player something the policy cannot see.
        visible_roles = list(anchors)[: engine.staff.coarse_availability() + 1]
        for role, (ax, ay) in anchors.items():
            if role not in visible_roles:
                continue
            wobble = ((self.frame // 12) + hash(role)) % 3 - 1
            x = (ax + wobble) * TILE
            y = ay * TILE
            if not engine.ward_map.walkable(ax + wobble, ay):
                x = ax * TILE
            sprite = self.sprites.people[role][(self.frame // 6) % 2]
            self.surface.blit(sprite, (x, y))

    def _draw_agent(self) -> None:
        engine = self.engine
        sprite = self.sprites.people["agent"][(self.frame // 5) % 2]
        x, y = engine.agent_x * TILE, engine.agent_y * TILE
        # A soft ring so the player never loses their character.
        pygame.draw.circle(
            self.surface, (255, 226, 120), (x + TILE // 2, y + TILE - 3), 7, width=2
        )
        self.surface.blit(sprite, (x, y))

    # ------------------------------------------------------------------
    def _draw_hud(self) -> None:
        engine = self.engine
        cfg = engine.cfg
        x0 = engine.ward_map.width * TILE
        panel = pygame.Rect(x0, 0, HUD_WIDTH, self.height)
        pygame.draw.rect(self.surface, PANEL_BG, panel)
        pygame.draw.line(self.surface, (60, 68, 84), (x0, 0), (x0, self.height))

        pad = 16
        y = 14

        minutes = engine.step_index * cfg.minutes_per_step
        clock = f"{7 + minutes // 60:02d}:{minutes % 60:02d}"
        clock_surface = self.font_big.render(clock, True, TEXT)
        self.surface.blit(clock_surface, (x0 + pad, y))
        shift_label = self.font_small.render(
            f"step {engine.step_index}/{cfg.steps_per_episode}", True, TEXT_DIM
        )
        # Right-aligned against the panel edge so it can never collide with the
        # clock, however wide either grows.
        self.surface.blit(
            shift_label,
            (x0 + HUD_WIDTH - pad - shift_label.get_width(),
             y + clock_surface.get_height() - shift_label.get_height() - 2),
        )
        y += clock_surface.get_height() + 10

        # Progress bar for the shift.
        bar = pygame.Rect(x0 + pad, y, HUD_WIDTH - 2 * pad, 5)
        pygame.draw.rect(self.surface, PANEL_ALT, bar, border_radius=3)
        frac = min(1.0, engine.step_index / max(1, cfg.steps_per_episode))
        pygame.draw.rect(
            self.surface,
            ACCENT,
            pygame.Rect(bar.x, bar.y, int(bar.width * frac), bar.height),
            border_radius=3,
        )
        y += 20

        y = self._draw_stats(x0, y, pad)
        y = self._draw_alarm_board(x0, y, pad)
        self._draw_footer(x0, pad)

    def _draw_stats(self, x0: int, y: int, pad: int) -> int:
        engine = self.engine
        flow = engine.flow
        stats = [
            ("Beds", f"{flow.occupied_beds}/{flow.n_beds}"),
            ("Free", str(flow.free_beds)),
            ("Queue", str(flow.queue_length)),
            ("Enrolled", str(sum(1 for p in flow.patients() if p.is_enrolled))),
        ]
        gap = 10
        box_w = (HUD_WIDTH - 2 * pad - gap) // 2
        box_h = self.line_small + self.line + 12
        for i, (label, value) in enumerate(stats):
            bx = x0 + pad + (i % 2) * (box_w + gap)
            by = y + (i // 2) * (box_h + gap)
            pygame.draw.rect(
                self.surface, PANEL_ALT, pygame.Rect(bx, by, box_w, box_h), border_radius=6
            )
            self.surface.blit(self.font_small.render(label, True, TEXT_DIM), (bx + 10, by + 5))
            self.surface.blit(
                self.font.render(value, True, TEXT), (bx + 10, by + 5 + self.line_small)
            )
        y += 2 * box_h + gap + 14

        staff_words = ("skeleton", "stretched", "comfortable")
        level = engine.staff.coarse_availability()
        self.surface.blit(
            self.font_small.render(f"Staff: {staff_words[level]}", True, TEXT_DIM),
            (x0 + pad, y),
        )
        y += self.line_small + 10
        return y

    def _draw_alarm_board(self, x0: int, y: int, pad: int) -> int:
        engine = self.engine
        title = "TELEMETRY" if engine.cfg.telemetry_enabled else "TELEMETRY OFF"
        self.surface.blit(self.font_small.render(title, True, ACCENT), (x0 + pad, y))
        y += self.line_small + 4

        if not engine.cfg.telemetry_enabled:
            note = self.font_small.render("no dashboard: check patients", True, TEXT_DIM)
            self.surface.blit(note, (x0 + pad, y))
            return y + self.line_small + 10

        alarms = engine.visible_alarms()
        if not alarms:
            at_station = engine.ward_map.at_station(engine.agent_x, engine.agent_y)
            lines = (
                ["no active alarms"]
                if at_station
                else ["press D to check the board", "(or stand at the station)"]
            )
            for line in lines:
                self.surface.blit(self.font_small.render(line, True, TEXT_DIM), (x0 + pad, y))
                y += self.line_small
            return y + 10

        alarms.sort(key=lambda a: (not a.is_urgent, -a.age(engine.step_index)))
        for alarm in alarms[:8]:
            colour = ALARM_COLOURS.get(alarm.kind, sprites.ALARM_URGENT)
            row = pygame.Rect(x0 + pad, y, HUD_WIDTH - 2 * pad, self.line_small + 8)
            pygame.draw.rect(self.surface, PANEL_ALT, row, border_radius=4)
            pygame.draw.rect(self.surface, colour, pygame.Rect(row.x, row.y, 4, row.height), border_radius=2)
            text = f"bed {alarm.bed:2d} {alarm.kind.value:11s} {alarm.cgm_value:4.1f}"
            self.surface.blit(self.font_small.render(text, True, TEXT), (row.x + 11, row.y + 4))
            age = self.font_small.render(f"+{alarm.age(engine.step_index)}", True, TEXT_DIM)
            self.surface.blit(age, (row.right - age.get_width() - 8, row.y + 4))
            y += row.height + 4
        return y + 8

    def _draw_footer(self, x0: int, pad: int) -> None:
        engine = self.engine
        available = HUD_WIDTH - 2 * pad

        def fit(text: str) -> str:
            """Trim to the panel width using measured text, not a guessed
            character count - the disclaimer was being clipped mid-word."""
            if self.font_small.size(text)[0] <= available:
                return text
            while text and self.font_small.size(text + "…")[0] > available:
                text = text[:-1]
            return text + "…"

        lines = [
            (fit(engine.last_action_result), TEXT_DIM),
            (fit(f"return {engine.rewards.total:+.1f}"), TEXT),
            (fit("academic model - not clinical advice"), (108, 114, 128)),
        ]
        y = self.height - pad - self.line_small * len(lines)
        for text, colour in lines:
            self.surface.blit(self.font_small.render(text, True, colour), (x0 + pad, y))
            y += self.line_small
