"""Ward artwork: the pixel-art sheets, with a procedural fallback.

Two sources, one API. `render/assets/` holds original CC0 pixel art - an
orthogonal top-down tileset, a 7-character sheet and a set of clinical status
overlays - described by `assets-index.json`. If that directory is absent the
same API is served by the procedural rectangles further down, so the simulator
still runs without the art. Nothing here is traced or recoloured from any
commercial game.

Skin tones and blanket colours are *indexed palette regions*, swapped at load
time rather than shipped as pre-rendered combinations: five skin tones and
eight blankets cost five and forty small surfaces instead of forty sheets.

Stdlib + pygame only (ships to the browser build).
"""

import json
from pathlib import Path

import pygame

# On-screen pixels per map tile. Larger than strictly necessary so the canvas
# is rendered at a high enough native resolution that the browser's upscale
# does not turn everything to mush.
TILE = 32

# The art is authored at 16px and displayed at 2x (integer, nearest-neighbour -
# any other factor would smear the pixel grid).
SOURCE_TILE = 16
SCALE = TILE // SOURCE_TILE

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Renderer role name -> character name in the sheet.
ROLE_CHARACTERS = {
    "agent": "ward_nurse_player",
    "hca": "healthcare_assistant",
    "nurse": "staff_nurse",
    "doctor": "doctor",
    "surgeon": "surgeon",
    "diabetes": "diabetes_specialist_nurse",
    "patient": "patient_walking",
}
DIRECTIONS = ("down", "left", "right", "up")
# The classic four-beat walk: contact, pass, contact, pass.
WALK_CYCLE = ("left_step", "idle", "right_step", "idle")

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
    viewer. See docs/ASSET_BRIEF.md; the sheets in assets/ follow the same rule
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


def _to_rgba(surface: pygame.Surface) -> pygame.Surface:
    """Copy into an explicit 32-bit RGBA surface.

    Not `convert_alpha()`: that adopts the *display* format and raises when
    there is no display, which is exactly the Gymnasium `rgb_array` path -
    `WardRenderer(headless=True)` never calls `set_mode`. Blitting onto a
    fresh SRCALPHA surface carries the colour-keyed transparency across
    without needing a display at all.
    """
    out = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
    out.blit(surface, (0, 0))
    return out


def _recolour(source: pygame.Surface, swaps) -> pygame.Surface:
    """Return a copy of `source` with the swapped palette regions applied.

    `swaps` is a sequence of (palette_index, baked_hex, replacement_hex).

    Two routes, because the browser's SDL_image build does not necessarily
    hand back an indexed surface the way the desktop one does. When the surface
    is still 8-bit we repaint the palette entry, which touches exactly the
    intended region. Otherwise we match on the baked colour - safe only because
    every opaque entry in the 48-colour palette is unique, which
    `tests/test_render_assets.py` pins down.
    """
    out = source.copy()
    if out.get_bitsize() == 8:
        for index, _baked, replacement in swaps:
            out.set_palette_at(index, pygame.Color(replacement))
        return _to_rgba(out)

    out = _to_rgba(out)
    if not swaps:
        return out

    pairs = [(pygame.Color(baked), pygame.Color(new)) for _i, baked, new in swaps]
    try:
        pixels = pygame.PixelArray(out)
    except (AttributeError, NotImplementedError, pygame.error):
        # Last resort for a runtime without PixelArray. Slow, but it only ever
        # runs on sheets a few thousand pixels across, and a slow ward beats a
        # blank one.
        width, height = out.get_size()
        lookup = {tuple(old): new for old, new in pairs}
        for x in range(width):
            for y in range(height):
                replacement = lookup.get(tuple(out.get_at((x, y))))
                if replacement is not None:
                    out.set_at((x, y), replacement)
        return out

    for old, new in pairs:
        pixels.replace(old, new)
    pixels.close()
    return out


class AssetPack:
    """The art sheets, sliced and scaled to TILE.

    Raises if the sheets are present but unreadable. That is deliberate: a
    missing `assets/` directory is the supported fallback path, but a corrupt
    one is a bug that should surface in the build, not degrade silently into
    rectangles that nobody notices until it is deployed.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        manifest = json.loads(
            (directory / "assets-index.json").read_text(encoding="utf-8")
        )
        sheets = manifest["sheets"]
        palette = manifest["palette"]
        self.skin = palette["skin_indices"]
        self.blanket = palette["blanket_indices"]
        self.n_skins = len(self.skin["variants"])
        self.n_blankets = len(self.blanket["variants"])

        self.tiles = self._named(sheets["tiles.png"], "tiles.png")
        self.bed_overlays = self._named(sheets["overlays_bed.png"], "overlays_bed.png")
        self.effects = self._named(sheets["overlays_effect.png"], "overlays_effect.png")
        self.characters = self._characters(sheets["characters.png"])
        self.patients = self._patients(sheets["patients_in_bed.png"])

    # -- loading helpers ---------------------------------------------------
    def _load(self, filename: str) -> pygame.Surface:
        return pygame.image.load(str(self.directory / filename))

    @staticmethod
    def _cut(sheet: pygame.Surface, item: dict) -> pygame.Surface:
        rect = pygame.Rect(item["x"], item["y"], item["width"], item["height"])
        piece = sheet.subsurface(rect).copy()
        if piece.get_bitsize() == 8:
            piece = _to_rgba(piece)
        # scale(), unlike smoothscale(), is nearest-neighbour - it keeps the
        # pixel grid hard, which is the whole point of pixel art.
        return pygame.transform.scale(
            piece, (item["width"] * SCALE, item["height"] * SCALE)
        )

    def _named(self, spec: dict, filename: str) -> dict:
        sheet = _to_rgba(self._load(filename))
        return {item["name"]: self._cut(sheet, item) for item in spec["items"]}

    # -- palette regions ---------------------------------------------------
    def _skin_swaps(self, index: int):
        variants = self.skin["variants"]
        baked, variant = variants[0], variants[index]
        if index == 0:
            return []
        return [
            (self.skin[part], baked[part], variant[part])
            for part in ("main", "detail", "shadow")
        ]

    def _blanket_swaps(self, index: int):
        variants = self.blanket["variants"]
        baked, variant = variants[0], variants[index]
        if index == 0:
            return []
        return [
            (self.blanket[part], baked[part], variant[part])
            for part in ("main", "shadow")
        ]

    # -- sheets ------------------------------------------------------------
    def _characters(self, spec: dict) -> dict:
        self._raw_characters = self._load("characters.png")
        self._character_items = spec["items"]
        self._patient_cache: dict = {}
        out = {}
        for skin in range(self.n_skins):
            sheet = _recolour(self._raw_characters, self._skin_swaps(skin))
            for item in spec["items"]:
                key = (item["character"], item["direction"], item["frame"], skin)
                out[key] = self._cut(sheet, item)
        return out

    def patient_frames(self, skin: int, blanket: int) -> dict:
        """Walking-patient frames whose gown trim matches their bed blanket.

        Built on demand rather than up front: 5 skins x 8 blankets x 12 frames
        would be 480 surfaces to cover the handful of patients who are out of
        bed at any moment. One recolour pass yields all twelve frames for a
        combination, and the result is cached for the rest of the shift.
        """
        key = (skin, blanket)
        frames = self._patient_cache.get(key)
        if frames is None:
            sheet = _recolour(
                self._raw_characters,
                self._skin_swaps(skin) + self._blanket_swaps(blanket),
            )
            frames = {
                (item["direction"], item["frame"]): self._cut(sheet, item)
                for item in self._character_items
                if item["character"] == "patient_walking"
            }
            self._patient_cache[key] = frames
        return frames

    def _patients(self, spec: dict) -> list:
        raw = self._load("patients_in_bed.png")
        item = spec["items"][0]
        bed = self.tiles["hospital_bed_made"]
        grid = []
        for skin in range(self.n_skins):
            row = []
            for blanket in range(self.n_blankets):
                swaps = self._skin_swaps(skin) + self._blanket_swaps(blanket)
                occupant = self._cut(_recolour(raw, swaps), item)
                composite = bed.copy()
                composite.blit(occupant, (0, 0))
                row.append(composite)
            grid.append(row)
        return grid


class SpriteSheet:
    """Ward artwork behind one API. Requires an initialised pygame display."""

    def __init__(self, assets_dir=None):
        directory = ASSETS_DIR if assets_dir is None else Path(assets_dir)
        self.pack = AssetPack(directory) if (directory / "assets-index.json").is_file() else None
        self.using_assets = self.pack is not None

        # A character is 16x24 on a 16x16 tile, so it stands a half-tile proud
        # of the tile it occupies and its feet land on the floor. The
        # rectangles are tile-sized and need no such offset.
        self.character_y_offset = -(24 - SOURCE_TILE) * SCALE if self.using_assets else 0

        if self.using_assets:
            self.n_skins = self.pack.n_skins
            self.n_blankets = self.pack.n_blankets
            self.patients = self.pack.patients
            return

        self.n_skins = len(SKIN_TONES)
        self.n_blankets = len(BLANKET_COLOURS)
        self._floor = [make_floor(0), make_floor(1)]
        self._wall = make_wall()
        self._bed = make_bed()
        self._station = make_station()
        self._drug_room = make_drug_room()
        self._entrance = make_entrance()
        self.patients = [
            [make_patient_in_bed(skin, blanket) for blanket in range(self.n_blankets)]
            for skin in range(self.n_skins)
        ]
        self._people = {
            role: [make_person(colour, i % len(SKIN_TONES), bob, role=role)
                   for i, bob in enumerate((0, 1))]
            for role, colour in STAFF_COLOURS.items()
        }
        self._people["patient"] = [
            make_person(PATIENT_GOWN, 1, bob, role="patient") for bob in (0, 1)
        ]

    # ------------------------------------------------------------------
    def tile(self, name: str) -> pygame.Surface:
        """A map tile by its name in the manifest."""
        if self.using_assets:
            return self.pack.tiles[name]
        # The rectangles have no per-name variants; map each family onto the
        # one shape that stands for it.
        if name.startswith("wall"):
            return self._wall
        if name.startswith("desk"):
            return self._station
        if name.startswith("drug_room"):
            return self._drug_room
        if name.startswith("ward_entrance") or name == "entrance_mat":
            return self._entrance
        if name.startswith("hospital_bed"):
            return self._bed
        return self._floor[1] if name == "ward_floor_mid" else self._floor[0]

    def person(self, role: str, direction: str = "down", phase: int = 0,
               skin: int = 0, blanket=None) -> pygame.Surface:
        """One character frame. Direction is ignored by the fallback.

        `blanket` applies to patients only: their gown trim carries the same
        colour as their bed blanket, so a patient walking to theatre is still
        recognisably the patient who was in bed 12.
        """
        if not self.using_assets:
            return self._people[role][phase % 2]
        frame = WALK_CYCLE[phase % len(WALK_CYCLE)]
        character = ROLE_CHARACTERS[role]
        if blanket is not None and character == "patient_walking":
            frames = self.pack.patient_frames(
                skin % self.n_skins, blanket % self.n_blankets
            )
            return frames[(direction, frame)]
        return self.pack.characters[(character, direction, frame, skin % self.n_skins)]

    def patient_in_bed(self, skin: int, blanket: int) -> pygame.Surface:
        return self.patients[skin % self.n_skins][blanket % self.n_blankets]

    def bed_overlay(self, name: str):
        """A clinical status overlay, or None when running on rectangles."""
        return self.pack.bed_overlays.get(name) if self.using_assets else None

    def effect(self, name: str):
        return self.pack.effects.get(name) if self.using_assets else None
