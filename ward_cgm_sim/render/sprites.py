"""Original pixel-art sprites, drawn in code at import time.

Everything here is generated procedurally from rectangles - there are no image
files and no third-party assets, so nothing in this repository infringes any
game's artwork. The look is the familiar top-down 16-bit ward: chunky tiles,
flat colours, a two-frame walk bob.

Stdlib + pygame only (ships to the browser build).
"""

import pygame

TILE = 24  # on-screen pixels per map tile

# --- palette -----------------------------------------------------------
FLOOR_A = (232, 226, 212)
FLOOR_B = (222, 215, 200)
WALL_DARK = (86, 96, 112)
WALL_LIGHT = (120, 132, 150)
BED_FRAME = (176, 182, 196)
BED_SHEET = (244, 246, 250)
BED_PILLOW = (214, 222, 236)
STATION_DESK = (128, 104, 78)
STATION_TOP = (158, 132, 100)
SCREEN_ON = (74, 168, 196)
DRUG_ROOM_COLOUR = (150, 176, 148)
ENTRANCE_COLOUR = (196, 186, 160)

SKIN_TONES = [
    (240, 205, 178),
    (222, 176, 140),
    (190, 140, 104),
    (150, 104, 72),
    (108, 74, 52),
]
NURSE_UNIFORM = (46, 108, 170)
HCA_UNIFORM = (120, 176, 200)
DOCTOR_UNIFORM = (240, 242, 246)
SURGEON_UNIFORM = (86, 150, 132)
DIABETES_UNIFORM = (150, 110, 178)
PATIENT_GOWN = (206, 214, 226)

ALARM_URGENT = (216, 68, 74)
ALARM_WARNING = (232, 158, 62)
ENROLLED_MARK = (72, 176, 128)
SIGNAL_LOST = (140, 140, 150)

STAFF_COLOURS = {
    "agent": NURSE_UNIFORM,
    "hca": HCA_UNIFORM,
    "nurse": (70, 140, 196),
    "doctor": DOCTOR_UNIFORM,
    "surgeon": SURGEON_UNIFORM,
    "diabetes": DIABETES_UNIFORM,
}


def _surface(size: int = TILE) -> pygame.Surface:
    return pygame.Surface((size, size), pygame.SRCALPHA)


def make_floor(variant: int = 0) -> pygame.Surface:
    surf = _surface()
    base = FLOOR_A if variant == 0 else FLOOR_B
    surf.fill(base)
    # Faint tile seam, the classic top-down floor grid.
    pygame.draw.line(surf, (0, 0, 0, 18), (0, TILE - 1), (TILE, TILE - 1))
    pygame.draw.line(surf, (0, 0, 0, 18), (TILE - 1, 0), (TILE - 1, TILE))
    return surf


def make_wall() -> pygame.Surface:
    surf = _surface()
    surf.fill(WALL_DARK)
    pygame.draw.rect(surf, WALL_LIGHT, pygame.Rect(0, 0, TILE, TILE // 3))
    pygame.draw.line(surf, (0, 0, 0, 60), (0, TILE // 3), (TILE, TILE // 3))
    return surf


def make_bed() -> pygame.Surface:
    surf = _surface()
    surf.fill(FLOOR_A)
    frame = pygame.Rect(2, 1, TILE - 4, TILE - 2)
    pygame.draw.rect(surf, BED_FRAME, frame, border_radius=3)
    sheet = pygame.Rect(3, 5, TILE - 6, TILE - 7)
    pygame.draw.rect(surf, BED_SHEET, sheet, border_radius=2)
    pillow = pygame.Rect(5, 2, TILE - 10, 4)
    pygame.draw.rect(surf, BED_PILLOW, pillow, border_radius=2)
    return surf


def make_station() -> pygame.Surface:
    surf = _surface()
    surf.fill(FLOOR_A)
    pygame.draw.rect(surf, STATION_DESK, pygame.Rect(1, 6, TILE - 2, TILE - 7), border_radius=2)
    pygame.draw.rect(surf, STATION_TOP, pygame.Rect(1, 4, TILE - 2, 4), border_radius=2)
    # The telemetry monitor.
    pygame.draw.rect(surf, (40, 46, 58), pygame.Rect(6, 8, TILE - 12, 8), border_radius=1)
    pygame.draw.rect(surf, SCREEN_ON, pygame.Rect(7, 9, TILE - 14, 6))
    return surf


def make_drug_room() -> pygame.Surface:
    surf = _surface()
    surf.fill(DRUG_ROOM_COLOUR)
    pygame.draw.rect(surf, (250, 250, 250), pygame.Rect(TILE // 2 - 5, 5, 10, 3))
    pygame.draw.rect(surf, (250, 250, 250), pygame.Rect(TILE // 2 - 2, 2, 4, 9))
    pygame.draw.rect(surf, (110, 140, 110), pygame.Rect(4, TILE - 9, TILE - 8, 7), border_radius=2)
    return surf


def make_entrance() -> pygame.Surface:
    surf = _surface()
    surf.fill(ENTRANCE_COLOUR)
    pygame.draw.rect(surf, (150, 140, 118), pygame.Rect(3, 2, TILE - 6, TILE - 4), border_radius=2)
    pygame.draw.line(surf, (110, 100, 84), (TILE // 2, 2), (TILE // 2, TILE - 2), 2)
    return surf


def make_person(body_colour, skin_index: int = 0, bob: int = 0) -> pygame.Surface:
    """A small two-tone character sprite with a walk bob.

    ``bob`` shifts the body a pixel to animate movement, the cheapest possible
    nod to the walking animation in a top-down RPG.
    """
    surf = _surface()
    skin = SKIN_TONES[skin_index % len(SKIN_TONES)]
    top = 4 + bob

    # head
    pygame.draw.rect(surf, skin, pygame.Rect(TILE // 2 - 4, top, 8, 7), border_radius=3)
    # hair / cap line
    pygame.draw.rect(surf, (58, 48, 44), pygame.Rect(TILE // 2 - 4, top, 8, 2), border_radius=2)
    # body
    pygame.draw.rect(surf, body_colour, pygame.Rect(TILE // 2 - 5, top + 7, 10, 9), border_radius=2)
    # arms
    pygame.draw.rect(surf, body_colour, pygame.Rect(TILE // 2 - 7, top + 8, 2, 6))
    pygame.draw.rect(surf, body_colour, pygame.Rect(TILE // 2 + 5, top + 8, 2, 6))
    # legs
    pygame.draw.rect(surf, (52, 58, 72), pygame.Rect(TILE // 2 - 4, top + 16, 3, 4))
    pygame.draw.rect(surf, (52, 58, 72), pygame.Rect(TILE // 2 + 1, top + 16, 3, 4))
    return surf


def make_patient_in_bed(skin_index: int = 0) -> pygame.Surface:
    """A patient lying in bed: head on the pillow, blanket over the body."""
    surf = make_bed()
    skin = SKIN_TONES[skin_index % len(SKIN_TONES)]
    pygame.draw.rect(surf, skin, pygame.Rect(TILE // 2 - 3, 2, 6, 5), border_radius=2)
    pygame.draw.rect(surf, PATIENT_GOWN, pygame.Rect(4, 8, TILE - 8, TILE - 11), border_radius=2)
    return surf


class SpriteSheet:
    """Lazily-built sprite cache. Requires an initialised pygame display."""

    def __init__(self):
        self.floor = [make_floor(0), make_floor(1)]
        self.wall = make_wall()
        self.bed = make_bed()
        self.station = make_station()
        self.drug_room = make_drug_room()
        self.entrance = make_entrance()
        self.patients = [make_patient_in_bed(i) for i in range(len(SKIN_TONES))]
        self.people = {
            role: [make_person(colour, i % len(SKIN_TONES), bob)
                   for i, bob in enumerate((0, 1))]
            for role, colour in STAFF_COLOURS.items()
        }
        self.walking_patient = [make_person(PATIENT_GOWN, 1, bob) for bob in (0, 1)]
