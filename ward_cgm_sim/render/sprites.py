"""Original pixel-art sprites, drawn in code at import time.

Everything here is generated procedurally from rectangles - there are no image
files and no third-party assets, so nothing in this repository infringes any
game's artwork. The look is the familiar top-down 16-bit ward: chunky tiles,
flat colours, a two-frame walk bob.

Stdlib + pygame only (ships to the browser build).
"""

import pygame

# On-screen pixels per map tile. Larger than strictly necessary so the canvas
# is rendered at a high enough native resolution that the browser's upscale
# does not turn everything to mush.
TILE = 32

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

# Per-patient blanket colours. A viewer following one patient across a shift
# identifies them by blanket, not by the tiny bed number - so these must be
# distinguishable from each other AND from every alarm colour. Red, amber,
# orange, yellow and green are deliberately absent: those mean clinical state.
BLANKET_COLOURS = [
    (122, 152, 186),  # dusty blue
    (140, 166, 140),  # sage
    (158, 142, 186),  # lilac
    (110, 164, 168),  # teal
    (166, 158, 148),  # warm grey
    (196, 154, 168),  # soft pink
    (188, 170, 134),  # sand
    (146, 182, 172),  # pale mint
]

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


def u(n: float) -> int:
    """Scale a length written against the original 24px tile to TILE."""
    return max(1, round(n * TILE / 24))


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
    frame = pygame.Rect(u(2), u(1), TILE - u(4), TILE - u(2))
    pygame.draw.rect(surf, BED_FRAME, frame, border_radius=u(3))
    sheet = pygame.Rect(u(3), u(5), TILE - u(6), TILE - u(7))
    pygame.draw.rect(surf, BED_SHEET, sheet, border_radius=u(2))
    pillow = pygame.Rect(u(5), u(2), TILE - u(10), u(4))
    pygame.draw.rect(surf, BED_PILLOW, pillow, border_radius=u(2))
    return surf


def make_station() -> pygame.Surface:
    surf = _surface()
    surf.fill(FLOOR_A)
    pygame.draw.rect(surf, STATION_DESK, pygame.Rect(u(1), u(6), TILE - u(2), TILE - u(7)), border_radius=u(2))
    pygame.draw.rect(surf, STATION_TOP, pygame.Rect(u(1), u(4), TILE - u(2), u(4)), border_radius=u(2))
    # The telemetry monitor.
    pygame.draw.rect(surf, (40, 46, 58), pygame.Rect(u(6), u(8), TILE - u(12), u(8)), border_radius=u(1))
    pygame.draw.rect(surf, SCREEN_ON, pygame.Rect(u(7), u(9), TILE - u(14), u(6)))
    return surf


def make_drug_room() -> pygame.Surface:
    surf = _surface()
    surf.fill(DRUG_ROOM_COLOUR)
    pygame.draw.rect(surf, (250, 250, 250), pygame.Rect(TILE // 2 - u(5), u(5), u(10), u(3)))
    pygame.draw.rect(surf, (250, 250, 250), pygame.Rect(TILE // 2 - u(2), u(2), u(4), u(9)))
    pygame.draw.rect(surf, (110, 140, 110), pygame.Rect(u(4), TILE - u(9), TILE - u(8), u(7)), border_radius=u(2))
    return surf


def make_entrance() -> pygame.Surface:
    surf = _surface()
    surf.fill(ENTRANCE_COLOUR)
    pygame.draw.rect(surf, (150, 140, 118), pygame.Rect(u(3), u(2), TILE - u(6), TILE - u(4)), border_radius=u(2))
    pygame.draw.line(surf, (110, 100, 84), (TILE // 2, u(2)), (TILE // 2, TILE - u(2)), u(2))
    return surf


def make_person(
    body_colour,
    skin_index: int = 0,
    bob: int = 0,
    role: str = "nurse",
) -> pygame.Surface:
    """A character sprite carrying a role-identifying attribute.

    Role is read from the ATTRIBUTE - cap, coat, mask - not from tunic colour,
    which is indistinguishable at this size and invisible to a colourblind
    viewer. See docs/ASSET_BRIEF.md; the commissioned art follows the same rule
    and these placeholders are a stand-in for it.
    """
    surf = _surface()
    skin = SKIN_TONES[skin_index % len(SKIN_TONES)]
    top = u(4) + bob
    cx = TILE // 2

    # --- torso, arms, legs ------------------------------------------------
    coat = role == "doctor"
    body_h = u(11) if coat else u(9)          # a doctor's coat flares lower
    body_w = u(12) if coat else u(10)
    pygame.draw.rect(
        surf, body_colour,
        pygame.Rect(cx - body_w // 2, top + u(7), body_w, body_h), border_radius=u(2)
    )
    pygame.draw.rect(surf, body_colour, pygame.Rect(cx - u(7), top + u(8), u(2), u(6)))
    pygame.draw.rect(surf, body_colour, pygame.Rect(cx + u(5), top + u(8), u(2), u(6)))
    leg_y = top + u(18) if coat else top + u(16)
    pygame.draw.rect(surf, (52, 58, 72), pygame.Rect(cx - u(4), leg_y, u(3), u(4)))
    pygame.draw.rect(surf, (52, 58, 72), pygame.Rect(cx + u(1), leg_y, u(3), u(4)))

    # --- head -------------------------------------------------------------
    pygame.draw.rect(surf, skin, pygame.Rect(cx - u(4), top, u(8), u(7)), border_radius=u(3))

    # --- the role attribute ----------------------------------------------
    if role in ("agent", "nurse", "diabetes"):
        # Nurse's cap: white, sitting proud of the head so it breaks the
        # silhouette. The player alone gets the red cross.
        pygame.draw.rect(surf, (250, 250, 252),
                         pygame.Rect(cx - u(5), top - u(1), u(10), u(3)), border_radius=u(1))
        if role == "agent":
            pygame.draw.rect(surf, ALARM_URGENT, pygame.Rect(cx - u(1), top - u(1), u(2), u(3)))
            pygame.draw.rect(surf, ALARM_URGENT, pygame.Rect(cx - u(2), top, u(4), u(1)))
    elif role == "doctor":
        # Dark hair plus a stethoscope loop at the collar.
        pygame.draw.rect(surf, (58, 48, 44), pygame.Rect(cx - u(4), top, u(8), u(2)), border_radius=u(2))
        pygame.draw.rect(surf, (40, 44, 54), pygame.Rect(cx - u(3), top + u(7), u(6), u(1)))
    elif role == "surgeon":
        # Scrub cap covering all hair, plus a mask band across the lower face.
        pygame.draw.rect(surf, SURGEON_UNIFORM,
                         pygame.Rect(cx - u(4), top - u(1), u(8), u(4)), border_radius=u(2))
        pygame.draw.rect(surf, (232, 238, 240), pygame.Rect(cx - u(4), top + u(4), u(8), u(2)))
    elif role == "hca":
        # Bare-headed, and carrying a clipboard - absence of a cap is the cue.
        pygame.draw.rect(surf, (92, 74, 58), pygame.Rect(cx - u(4), top, u(8), u(2)), border_radius=u(2))
        pygame.draw.rect(surf, (226, 220, 200), pygame.Rect(cx + u(2), top + u(10), u(4), u(5)))

    if role == "diabetes":
        # Visiting specialist: a bag strap crossing the torso.
        pygame.draw.line(surf, (68, 58, 74),
                         (cx - u(5), top + u(8)), (cx + u(4), top + u(14)), u(2))
    return surf


def make_patient_in_bed(skin_index: int = 0, blanket_index: int = 0) -> pygame.Surface:
    """A patient in bed, identified by blanket colour.

    With up to 32 patients on screen the printed bed number is too small to
    track at a glance, so the blanket carries the identity instead.
    """
    surf = make_bed()
    skin = SKIN_TONES[skin_index % len(SKIN_TONES)]
    blanket = BLANKET_COLOURS[blanket_index % len(BLANKET_COLOURS)]
    pygame.draw.rect(surf, skin, pygame.Rect(TILE // 2 - u(3), u(2), u(6), u(5)), border_radius=u(2))
    pygame.draw.rect(
        surf, blanket, pygame.Rect(u(4), u(8), TILE - u(8), TILE - u(11)), border_radius=u(2)
    )
    # A turned-back sheet edge, so the blanket reads as bedding, not a block.
    pygame.draw.rect(surf, BED_SHEET, pygame.Rect(u(4), u(8), TILE - u(8), u(2)))
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
        # One sprite per (skin tone, blanket colour) pair so a patient keeps a
        # stable identity for the whole shift.
        self.patients = [
            [make_patient_in_bed(skin, blanket) for blanket in range(len(BLANKET_COLOURS))]
            for skin in range(len(SKIN_TONES))
        ]
        self.people = {
            role: [make_person(colour, i % len(SKIN_TONES), bob, role=role)
                   for i, bob in enumerate((0, 1))]
            for role, colour in STAFF_COLOURS.items()
        }
        self.walking_patient = [
            make_person(PATIENT_GOWN, 1, bob, role="patient") for bob in (0, 1)
        ]
