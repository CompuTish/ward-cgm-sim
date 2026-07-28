#!/usr/bin/env python3
"""Draw the ward CGM pixel-art asset set from scratch.

The output is deliberately deterministic: every pixel is placed on the native
16 px grid, all sheets share one indexed palette, and the only alpha values are
fully transparent or fully opaque. Pillow is a development-time tool only; the
runtime continues to require pygame-ce alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ward_cgm_sim" / "render" / "assets"
QA = ROOT / "docs" / "asset-qa"

TILE = 16
CHAR_W = 16
CHAR_H = 24

# One shared 48-colour palette. Index 0 is transparent in every sheet.
PALETTE = [
    ("transparent", "#000000"),
    ("floor_light", "#E8E2D4"),
    ("floor_mid", "#DED7C8"),
    ("floor_scuff", "#CFC7B7"),
    ("corridor_cool", "#D6DCE2"),
    ("threshold", "#C5BCA9"),
    ("wall_dark", "#566070"),
    ("wall_light", "#788496"),
    ("wall_top", "#AAB3C0"),
    ("outline", "#3F4956"),
    ("bed_frame", "#B0B6C4"),
    ("bed_linen", "#F4F6FA"),
    ("pillow", "#D6DEEC"),
    ("desk_wood", "#806A4E"),
    ("desk_top", "#9E8464"),
    ("screen_dark", "#28333D"),
    ("screen_glow", "#4AA8C4"),
    ("alarm_red", "#D8443A"),
    ("alarm_amber", "#E89E3E"),
    ("alarm_yellow", "#E7C84F"),
    ("ok_green", "#48B080"),
    ("inactive_grey", "#8C8C96"),
    ("navy", "#2E5B86"),
    ("mid_blue", "#468CC4"),
    ("pale_teal", "#78B0C8"),
    ("surgeon_teal", "#569684"),
    ("purple", "#966EB2"),
    ("patient_gown", "#CED6E2"),
    ("skin_main", "#F0CDB2"),
    ("hair_dark", "#3A302C"),
    ("hair_brown", "#5C4A3A"),
    ("hair_gold", "#B08A56"),
    ("trouser", "#343A48"),
    ("white", "#FAFAFC"),
    ("mask", "#E8EEF0"),
    ("deep_neutral", "#242A32"),
    ("blanket_main", "#7A98BA"),
    ("blanket_shadow", "#657F9E"),
    ("soft_pink", "#C49AA8"),
    ("sand", "#BCAA86"),
    ("pale_mint", "#92B6AC"),
    ("lilac", "#9E8EBA"),
    ("sage", "#8CA68C"),
    ("blanket_teal", "#6EA4A8"),
    ("door", "#6D7B87"),
    ("shadow", "#A9A49A"),
    ("skin_detail", "#E0B698"),
    ("skin_shadow", "#D8AD8F"),
]

IDX = {name: i for i, (name, _) in enumerate(PALETTE)}

SKIN_TONES = [
    {"name": "light", "main": "#F0CDB2", "detail": "#E0B698", "shadow": "#D8AD8F"},
    {"name": "light_medium", "main": "#DEB08C", "detail": "#CE9C78", "shadow": "#C4906B"},
    {"name": "medium", "main": "#BE8C68", "detail": "#AE7B58", "shadow": "#A46E4D"},
    {"name": "medium_dark", "main": "#966848", "detail": "#85593D", "shadow": "#7A4F35"},
    {"name": "dark", "main": "#6C4A34", "detail": "#5E3E2C", "shadow": "#503525"},
]

BLANKET_COLOURS = [
    {"name": "dusty_blue", "main": "#7A98BA", "shadow": "#657F9E"},
    {"name": "sage", "main": "#8CA68C", "shadow": "#718A71"},
    {"name": "lilac", "main": "#9E8EBA", "shadow": "#8172A0"},
    {"name": "teal", "main": "#6EA4A8", "shadow": "#56868A"},
    {"name": "warm_grey", "main": "#A69E94", "shadow": "#888178"},
    {"name": "soft_pink", "main": "#C49AA8", "shadow": "#A57D8A"},
    {"name": "sand", "main": "#BCAA86", "shadow": "#9C8B6A"},
    {"name": "pale_mint", "main": "#92B6AC", "shadow": "#75978E"},
]

TILE_NAMES = [
    "ward_floor_light",
    "ward_floor_mid",
    "ward_floor_scuff",
    "corridor_floor",
    "bay_threshold",
    "entrance_mat",
    "wall_horizontal_front",
    "wall_horizontal_top_edge",
    "wall_vertical_left",
    "wall_vertical_right",
    "wall_corner_outer_top_left",
    "wall_corner_outer_top_right",
    "wall_corner_inner",
    "wall_window_panel",
    "hospital_bed_made",
    "hospital_bed_disturbed",
    "bedside_cabinet",
    "iv_drip_stand",
    "curtain_open",
    "curtain_closed",
    "visitor_chair",
    "hand_wash_basin",
    "alcohol_gel_dispenser",
    "clinical_waste_bin",
    "desk_left_end",
    "desk_middle_run",
    "desk_right_end",
    "desk_monitor_on",
    "desk_monitor_alarm",
    "desk_keyboard_notes",
    "drug_room_door_closed",
    "drug_room_floor",
    "ward_entrance_double_closed",
    "ward_entrance_double_open",
    "store_cupboard",
    "sluice_dirty_utility_door",
]

CHARACTERS = [
    "ward_nurse_player",
    "healthcare_assistant",
    "staff_nurse",
    "doctor",
    "surgeon",
    "diabetes_specialist_nurse",
    "patient_walking",
]

DIRECTIONS = ["down", "left", "right", "up"]
FRAMES = ["left_step", "idle", "right_step"]

BED_OVERLAYS = [
    "sensor_working",
    "sensor_signal_lost",
    "alarm_hypoglycaemia",
    "alarm_severe_hypoglycaemia",
    "alarm_hyperglycaemia",
    "alarm_rapid_fall",
    "alarm_rapid_rise",
    "point_of_care_test",
    "treatment_given",
    "ready_for_discharge",
    "selected_adjacent_bed",
]


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _palette_bytes() -> list[int]:
    values: list[int] = []
    for _, colour in PALETTE:
        values.extend(_rgb(colour))
    values.extend([0] * (768 - len(values)))
    return values


def canvas(size: tuple[int, int], fill: int = 0) -> Image.Image:
    image = Image.new("P", size, fill)
    image.putpalette(_palette_bytes())
    image.info["transparency"] = 0
    return image


def save_indexed(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", transparency=0, optimize=False)


def rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], colour: str) -> None:
    draw.rectangle(box, fill=IDX[colour])


def line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    colour: str,
    width: int = 1,
) -> None:
    draw.line(points, fill=IDX[colour], width=width)


def polygon(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], colour: str) -> None:
    draw.polygon(points, fill=IDX[colour])


def _floor(fill: str = "floor_light") -> Image.Image:
    return canvas((TILE, TILE), IDX[fill])


def draw_bed(disturbed: bool = False) -> Image.Image:
    image = canvas((TILE, TILE))
    d = ImageDraw.Draw(image)
    rect(d, (2, 1, 13, 14), "outline")
    rect(d, (3, 1, 12, 13), "bed_frame")
    rect(d, (4, 3, 11, 12), "bed_linen")
    rect(d, (5, 3, 10, 5), "pillow")
    rect(d, (4, 12, 11, 13), "bed_frame")
    rect(d, (3, 14, 5, 15), "outline")
    rect(d, (10, 14, 12, 15), "outline")
    if disturbed:
        line(d, [(4, 8), (6, 7), (8, 9), (11, 7)], "pillow")
        line(d, [(4, 11), (7, 10), (9, 11), (11, 10)], "bed_frame")
    return image


def draw_tiles() -> Image.Image:
    tiles: list[Image.Image] = []

    # Floors 1-6
    tiles.append(_floor("floor_light"))
    tiles.append(_floor("floor_mid"))
    tile = _floor("floor_light")
    d = ImageDraw.Draw(tile)
    line(d, [(2, 11), (5, 11), (6, 10)], "floor_scuff")
    rect(d, (12, 3, 13, 3), "floor_scuff")
    tiles.append(tile)
    tile = _floor("corridor_cool")
    d = ImageDraw.Draw(tile)
    # Grout along two edges, so tiling produces a grid of squares the same size
    # as the bay tiles. A seam across the middle instead joins up into
    # continuous horizontal stripes and the corridor reads as floorboards.
    line(d, [(0, 15), (15, 15)], "floor_mid")
    line(d, [(15, 0), (15, 15)], "floor_mid")
    tiles.append(tile)
    tile = _floor("threshold")
    d = ImageDraw.Draw(tile)
    rect(d, (0, 6, 15, 9), "floor_mid")
    line(d, [(0, 6), (15, 6)], "wall_top")
    line(d, [(0, 9), (15, 9)], "floor_scuff")
    tiles.append(tile)
    tile = _floor("floor_light")
    d = ImageDraw.Draw(tile)
    rect(d, (1, 3, 14, 13), "outline")
    rect(d, (2, 4, 13, 12), "door")
    for y in (5, 8, 11):
        line(d, [(3, y), (12, y)], "wall_light")
    tiles.append(tile)

    # Walls 7-14
    tile = canvas((TILE, TILE), IDX["wall_dark"])
    d = ImageDraw.Draw(tile)
    rect(d, (0, 0, 15, 4), "wall_top")
    rect(d, (0, 5, 15, 11), "wall_light")
    line(d, [(0, 5), (15, 5)], "outline")
    rect(d, (0, 12, 15, 15), "wall_dark")
    tiles.append(tile)
    tile = canvas((TILE, TILE), IDX["floor_light"])
    d = ImageDraw.Draw(tile)
    rect(d, (0, 0, 15, 5), "wall_top")
    line(d, [(0, 5), (15, 5)], "outline")
    tiles.append(tile)
    for side in ("left", "right"):
        tile = canvas((TILE, TILE), IDX["floor_light"])
        d = ImageDraw.Draw(tile)
        if side == "left":
            rect(d, (0, 0, 6, 15), "wall_top")
            rect(d, (7, 0, 10, 15), "wall_light")
            rect(d, (11, 0, 12, 15), "wall_dark")
        else:
            rect(d, (9, 0, 15, 15), "wall_top")
            rect(d, (5, 0, 8, 15), "wall_light")
            rect(d, (3, 0, 4, 15), "wall_dark")
        tiles.append(tile)
    for side in ("left", "right"):
        tile = canvas((TILE, TILE), IDX["floor_light"])
        d = ImageDraw.Draw(tile)
        if side == "left":
            rect(d, (0, 0, 15, 5), "wall_top")
            rect(d, (0, 5, 6, 15), "wall_top")
            rect(d, (7, 5, 10, 15), "wall_light")
            rect(d, (0, 6, 6, 10), "wall_light")
        else:
            rect(d, (0, 0, 15, 5), "wall_top")
            rect(d, (9, 5, 15, 15), "wall_top")
            rect(d, (5, 5, 8, 15), "wall_light")
            rect(d, (9, 6, 15, 10), "wall_light")
        tiles.append(tile)
    tile = canvas((TILE, TILE), IDX["wall_top"])
    d = ImageDraw.Draw(tile)
    rect(d, (5, 5, 15, 15), "floor_light")
    rect(d, (0, 6, 4, 10), "wall_light")
    rect(d, (5, 0, 9, 4), "wall_light")
    rect(d, (4, 4, 5, 5), "outline")
    tiles.append(tile)
    tile = canvas((TILE, TILE), IDX["wall_dark"])
    d = ImageDraw.Draw(tile)
    rect(d, (0, 0, 15, 3), "wall_top")
    rect(d, (0, 4, 15, 13), "wall_light")
    rect(d, (2, 6, 13, 11), "outline")
    rect(d, (3, 7, 12, 10), "screen_glow")
    line(d, [(7, 7), (7, 10)], "white")
    tiles.append(tile)

    # Ward furniture 15-24
    tiles.append(draw_bed(False))
    tiles.append(draw_bed(True))
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (3, 5, 12, 14), "outline")
    rect(d, (4, 4, 11, 13), "desk_wood")
    rect(d, (4, 4, 11, 6), "desk_top")
    line(d, [(4, 9), (11, 9)], "outline")
    rect(d, (7, 7, 8, 7), "bed_frame")
    rect(d, (7, 11, 8, 11), "bed_frame")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (7, 2, 8, 13), "outline")
    rect(d, (6, 1, 9, 3), "bed_frame")
    rect(d, (5, 3, 10, 7), "white")
    rect(d, (6, 4, 9, 6), "screen_glow")
    line(d, [(7, 13), (4, 15)], "outline")
    line(d, [(8, 13), (11, 15)], "outline")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (1, 2, 14, 3), "bed_frame")
    rect(d, (2, 3, 3, 14), "inactive_grey")
    rect(d, (12, 3, 13, 14), "inactive_grey")
    rect(d, (4, 4, 5, 12), "pale_teal")
    rect(d, (10, 4, 11, 12), "pale_teal")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (1, 1, 14, 2), "bed_frame")
    rect(d, (2, 2, 13, 14), "pale_teal")
    for x in (4, 8, 12):
        line(d, [(x, 3), (x, 13)], "screen_glow")
    line(d, [(2, 14), (13, 14)], "outline")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (3, 4, 12, 12), "outline")
    rect(d, (4, 4, 11, 10), "desk_wood")
    rect(d, (5, 11, 6, 15), "outline")
    rect(d, (9, 11, 10, 15), "outline")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (2, 3, 13, 10), "outline")
    rect(d, (3, 3, 12, 8), "white")
    rect(d, (5, 5, 10, 7), "pillow")
    rect(d, (6, 10, 9, 12), "door")
    rect(d, (4, 13, 11, 14), "outline")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (5, 2, 10, 11), "outline")
    rect(d, (6, 3, 9, 9), "white")
    rect(d, (7, 4, 8, 6), "screen_glow")
    rect(d, (4, 11, 11, 13), "wall_light")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (3, 4, 12, 13), "outline")
    rect(d, (4, 5, 11, 12), "door")
    rect(d, (5, 3, 10, 5), "inactive_grey")
    rect(d, (5, 8, 10, 9), "wall_dark")
    rect(d, (6, 14, 9, 15), "outline")
    tiles.append(tile)

    # Nurse station 25-30
    for end in ("left", "middle", "right"):
        tile = canvas((TILE, TILE))
        d = ImageDraw.Draw(tile)
        rect(d, (0, 5, 15, 15), "outline")
        rect(d, (0, 5, 15, 8), "desk_top")
        rect(d, (0, 9, 15, 14), "desk_wood")
        if end == "left":
            rect(d, (0, 5, 1, 15), "outline")
            rect(d, (3, 11, 4, 12), "bed_frame")
        elif end == "right":
            rect(d, (14, 5, 15, 15), "outline")
            rect(d, (11, 11, 12, 12), "bed_frame")
        tiles.append(tile)
    for alarm in (False, True):
        tile = canvas((TILE, TILE))
        d = ImageDraw.Draw(tile)
        rect(d, (0, 7, 15, 15), "outline")
        rect(d, (0, 8, 15, 10), "desk_top")
        rect(d, (0, 11, 15, 14), "desk_wood")
        rect(d, (3, 1, 12, 8), "deep_neutral")
        rect(d, (4, 2, 11, 6), "alarm_red" if alarm else "screen_glow")
        if alarm:
            polygon(d, [(7, 2), (10, 5), (8, 5), (8, 7), (6, 7), (6, 5), (4, 5)], "white")
        else:
            line(d, [(4, 5), (6, 4), (7, 5), (9, 3), (11, 4)], "white")
        tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (0, 5, 15, 15), "outline")
    rect(d, (0, 5, 15, 8), "desk_top")
    rect(d, (0, 9, 15, 14), "desk_wood")
    rect(d, (3, 7, 9, 10), "deep_neutral")
    for x in (4, 6, 8):
        rect(d, (x, 8, x, 8), "white")
    rect(d, (10, 5, 14, 8), "bed_linen")
    line(d, [(11, 6), (13, 6)], "mid_blue")
    tiles.append(tile)

    # Rooms and doors 31-36
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (1, 0, 14, 15), "outline")
    rect(d, (2, 1, 13, 15), "door")
    rect(d, (3, 2, 12, 4), "wall_light")
    rect(d, (10, 6, 12, 10), "deep_neutral")
    rect(d, (11, 7, 11, 7), "screen_glow")
    rect(d, (4, 7, 7, 10), "white")
    rect(d, (5, 6, 6, 11), "white")
    tiles.append(tile)
    tile = _floor("corridor_cool")
    d = ImageDraw.Draw(tile)
    rect(d, (2, 2, 13, 4), "wall_light")
    for x in (3, 7, 11):
        rect(d, (x, 8, x + 1, 12), "bed_frame")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (0, 0, 15, 15), "outline")
    rect(d, (1, 1, 7, 15), "door")
    rect(d, (8, 1, 14, 15), "door")
    line(d, [(7, 1), (7, 15)], "wall_light")
    rect(d, (5, 8, 6, 9), "bed_frame")
    rect(d, (9, 8, 10, 9), "bed_frame")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (0, 0, 2, 15), "outline")
    rect(d, (13, 0, 15, 15), "outline")
    rect(d, (1, 1, 3, 13), "door")
    rect(d, (12, 1, 14, 13), "door")
    line(d, [(4, 14), (11, 14)], "threshold")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (2, 1, 13, 15), "outline")
    rect(d, (3, 2, 12, 15), "desk_wood")
    line(d, [(8, 2), (8, 15)], "outline")
    rect(d, (6, 8, 6, 8), "bed_frame")
    rect(d, (10, 8, 10, 8), "bed_frame")
    tiles.append(tile)
    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (1, 0, 14, 15), "outline")
    rect(d, (2, 1, 13, 15), "wall_light")
    rect(d, (3, 2, 12, 5), "bed_linen")
    line(d, [(4, 3), (11, 3)], "mid_blue")
    rect(d, (10, 8, 12, 10), "deep_neutral")
    rect(d, (11, 9, 11, 9), "inactive_grey")
    tiles.append(tile)

    assert len(tiles) == len(TILE_NAMES) == 36
    sheet = canvas((6 * TILE, 6 * TILE))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 6) * TILE, (i // 6) * TILE))
    return sheet


def _body_bounds(role: str, direction: str, bob: int) -> tuple[int, int, int, int]:
    if role == "doctor":
        return (3, 10 + bob, 12, 20 + bob)
    if role == "patient_walking":
        return (4, 12 + bob, 11, 20 + bob)
    if direction in ("left", "right"):
        return (4, 10 + bob, 11, 18 + bob)
    return (4, 10 + bob, 11, 18 + bob)


def draw_character(role: str, direction: str, frame: str) -> Image.Image:
    image = canvas((CHAR_W, CHAR_H))
    d = ImageDraw.Draw(image)
    step = -1 if frame == "left_step" else 1 if frame == "right_step" else 0
    bob = -1 if step else 0
    facing_left = direction == "left"
    facing_right = direction == "right"

    body_colour = {
        "ward_nurse_player": "navy",
        "healthcare_assistant": "pale_teal",
        "staff_nurse": "mid_blue",
        "doctor": "white",
        "surgeon": "surgeon_teal",
        "diabetes_specialist_nurse": "purple",
        "patient_walking": "patient_gown",
    }[role]

    # Feet and lower legs. All sprites touch y=23 for bottom-centre anchoring.
    if role == "patient_walking":
        leg_colour = "skin_main"
    elif role == "doctor":
        leg_colour = "trouser"
    else:
        leg_colour = body_colour if role in ("surgeon",) else "trouser"
    left_foot_x = 4 + (1 if step > 0 else 0)
    right_foot_x = 9 - (1 if step < 0 else 0)
    rect(d, (left_foot_x, 19 + bob, left_foot_x + 2, 22), leg_colour)
    rect(d, (right_foot_x, 19 + bob, right_foot_x + 2, 22), leg_colour)
    rect(d, (left_foot_x - 1, 22, left_foot_x + 2, 23), "outline")
    rect(d, (right_foot_x, 22, right_foot_x + 3, 23), "outline")
    if role == "patient_walking":
        rect(d, (left_foot_x, 19 + bob, left_foot_x + 2, 22), "skin_main")
        rect(d, (right_foot_x, 19 + bob, right_foot_x + 2, 22), "skin_main")
        rect(d, (left_foot_x - 1, 22, left_foot_x + 2, 23), "skin_shadow")
        rect(d, (right_foot_x, 22, right_foot_x + 3, 23), "skin_shadow")

    # Body silhouette and arms.
    bx0, by0, bx1, by1 = _body_bounds(role, direction, bob)
    rect(d, (bx0 - 1, by0, bx1 + 1, by1), "outline")
    if role == "doctor":
        polygon(d, [(4, by0), (11, by0), (12, by1), (3, by1)], body_colour)
        line(d, [(7, by0 + 1), (7, by1)], "bed_frame")
    else:
        rect(d, (bx0, by0, bx1, by1), body_colour)
    arm_y = 11 + bob
    if role != "patient_walking":
        rect(d, (2, arm_y, 4, arm_y + 6), "outline")
        rect(d, (11, arm_y, 13, arm_y + 6), "outline")
        rect(d, (3, arm_y, 4, arm_y + 5), body_colour)
        rect(d, (11, arm_y, 12, arm_y + 5), body_colour)
        rect(d, (3, arm_y + 6, 4, arm_y + 7), "skin_main")
        rect(d, (11, arm_y + 6, 12, arm_y + 7), "skin_main")

    # Head, hair and directional face.
    head_y = (5 if role == "patient_walking" else 3) + bob
    head_x0, head_x1 = (5, 10) if direction in ("left", "right") else (4, 11)
    rect(d, (head_x0, head_y, head_x1, head_y + 6), "skin_main")
    if direction == "up":
        rect(d, (head_x0, head_y, head_x1, head_y + 4), "hair_brown")
        rect(d, (head_x0 + 1, head_y + 5, head_x1 - 1, head_y + 6), "skin_shadow")
    else:
        rect(d, (head_x0, head_y, head_x1, head_y + 1), "hair_brown")
        if facing_left:
            rect(d, (head_x0, head_y + 1, head_x0 + 1, head_y + 4), "hair_brown")
            rect(d, (head_x0, head_y + 4, head_x0, head_y + 4), "skin_shadow")
        elif facing_right:
            rect(d, (head_x1 - 1, head_y + 1, head_x1, head_y + 4), "hair_brown")
            rect(d, (head_x1, head_y + 4, head_x1, head_y + 4), "skin_shadow")
        else:
            rect(d, (head_x0, head_y + 1, head_x0, head_y + 4), "hair_brown")
            rect(d, (head_x1, head_y + 1, head_x1, head_y + 4), "hair_brown")
            rect(d, (6, head_y + 4, 6, head_y + 4), "deep_neutral")
            rect(d, (9, head_y + 4, 9, head_y + 4), "deep_neutral")
            rect(d, (7, head_y + 5, 8, head_y + 5), "skin_detail")
        rect(d, (head_x0 + 1, head_y + 6, head_x1 - 1, head_y + 6), "skin_shadow")

    # Role-defining attributes.
    if role == "ward_nurse_player":
        # Tall cross-cap plus a longer flared tunic make the player unique.
        polygon(d, [(3, 14 + bob), (12, 14 + bob), (13, 20 + bob), (2, 20 + bob)], "outline")
        polygon(d, [(4, 14 + bob), (11, 14 + bob), (12, 19 + bob), (3, 19 + bob)], "navy")
        rect(d, (2, 3, 13, 4), "outline")
        rect(d, (3, 3, 12, 3), "white")
        rect(d, (5, 0, 10, 3), "outline")
        rect(d, (6, 0, 9, 2), "white")
        if direction != "up":
            rect(d, (7, 0, 8, 3), "mid_blue")
            rect(d, (6, 1, 9, 1), "mid_blue")
            line(d, [(7, 11 + bob), (7, 15 + bob)], "white")
            rect(d, (6, 15 + bob, 8, 17 + bob), "white")
            rect(d, (7, 16 + bob, 7, 16 + bob), "screen_glow")
    elif role == "staff_nurse":
        # Lower plain cap: same family, deliberately less prominent.
        rect(d, (4, head_y - 2, 11, head_y), "outline")
        rect(d, (5, head_y - 2, 10, head_y - 1), "white")
    elif role == "doctor":
        # Flared long coat plus a dark stethoscope loop.
        if direction != "up":
            line(d, [(5, 10 + bob), (5, 13 + bob), (7, 15 + bob), (9, 13 + bob), (9, 10 + bob)], "deep_neutral")
            rect(d, (4, 13 + bob, 5, 14 + bob), "deep_neutral")
        rect(d, (2, 17 + bob, 3, 20 + bob), "white")
        rect(d, (12, 17 + bob, 13, 20 + bob), "white")
    elif role == "surgeon":
        # Smooth all-hair scrub cap and mask band.
        polygon(d, [(2, 0), (13, 0), (15, 3), (13, 5), (2, 5), (0, 3)], "outline")
        polygon(d, [(3, 1), (12, 1), (13, 3), (12, 4), (3, 4), (2, 3)], "surgeon_teal")
        if direction != "up":
            rect(d, (4, head_y + 4, 11, head_y + 6), "mask")
            rect(d, (3, head_y + 4, 3, head_y + 5), "white")
            rect(d, (12, head_y + 4, 12, head_y + 5), "white")
    elif role == "healthcare_assistant":
        # No cap, short sleeves, wide tray/clipboard carried out front.
        rect(d, (2, 11 + bob, 4, 13 + bob), "skin_main")
        rect(d, (11, 11 + bob, 13, 13 + bob), "skin_main")
        if direction == "down":
            rect(d, (0, 14 + bob, 15, 17 + bob), "outline")
            rect(d, (1, 14 + bob, 14, 16 + bob), "bed_frame")
            rect(d, (5, 13 + bob, 10, 14 + bob), "white")
        elif direction == "left":
            rect(d, (0, 14 + bob, 7, 17 + bob), "outline")
            rect(d, (1, 14 + bob, 6, 16 + bob), "bed_frame")
        elif direction == "right":
            rect(d, (8, 14 + bob, 15, 17 + bob), "outline")
            rect(d, (9, 14 + bob, 14, 16 + bob), "bed_frame")
        else:
            rect(d, (4, 14 + bob, 11, 16 + bob), "outline")
    elif role == "diabetes_specialist_nurse":
        rect(d, (4, head_y - 2, 11, head_y), "outline")
        rect(d, (5, head_y - 2, 10, head_y - 1), "white")
        if direction in ("down", "up"):
            line(d, [(4, 10 + bob), (11, 18 + bob)], "deep_neutral", 2)
            rect(d, (11, 13 + bob, 15, 22), "outline")
            rect(d, (12, 14 + bob, 15, 21), "navy")
        elif direction == "left":
            line(d, [(10, 10 + bob), (5, 18 + bob)], "deep_neutral", 2)
            rect(d, (0, 13 + bob, 4, 22), "outline")
            rect(d, (0, 14 + bob, 3, 21), "navy")
        else:
            line(d, [(5, 10 + bob), (10, 18 + bob)], "deep_neutral", 2)
            rect(d, (11, 13 + bob, 15, 22), "outline")
            rect(d, (12, 14 + bob, 15, 21), "navy")
    elif role == "patient_walking":
        # Short stooped stance, gown trim, and an IV stand in one hand.
        rect(d, (3, 11 + bob, 12, 12 + bob), "outline")
        rect(d, (4, 11 + bob, 11, 12 + bob), "patient_gown")
        line(d, [(4, 18 + bob), (11, 18 + bob)], "blanket_main")
        if direction == "up":
            # Seen from behind, so the open back of the gown shows: a shadowed
            # opening down the spine with a tie at each end, as the brief asks.
            # Interior pixels only, so the silhouette is unchanged.
            rect(d, (7, 13 + bob, 8, 17 + bob), "deep_neutral")
            rect(d, (6, 14 + bob, 9, 14 + bob), "deep_neutral")
            rect(d, (6, 17 + bob, 9, 17 + bob), "deep_neutral")
        stand_x = 14 if direction != "right" else 1
        rect(d, (stand_x, 3, stand_x, 21), "outline")
        rect(d, (stand_x - 1, 3, stand_x + 1, 4), "bed_frame")
        rect(d, (stand_x - 2, 5, stand_x, 9), "white")
        rect(d, (stand_x - 1, 6, stand_x, 8), "screen_glow")
        line(d, [(stand_x, 21), (stand_x - 2, 23)], "outline")
        line(d, [(stand_x, 21), (stand_x + 1, 23)], "outline")

    return image


def draw_characters() -> Image.Image:
    sheet = canvas((3 * CHAR_W, len(CHARACTERS) * 4 * CHAR_H))
    for role_index, role in enumerate(CHARACTERS):
        for direction_index, direction in enumerate(DIRECTIONS):
            row = role_index * 4 + direction_index
            for frame_index, frame in enumerate(FRAMES):
                sprite = draw_character(role, direction, frame)
                sheet.paste(sprite, (frame_index * CHAR_W, row * CHAR_H))
    return sheet


def draw_patient_in_bed() -> Image.Image:
    image = canvas((TILE, TILE))
    d = ImageDraw.Draw(image)
    # Overlay only: the bed frame, pillow and linen live in tile 15.
    rect(d, (5, 3, 10, 5), "skin_shadow")
    rect(d, (5, 2, 10, 4), "skin_main")
    rect(d, (5, 2, 10, 2), "hair_brown")
    rect(d, (6, 4, 6, 4), "deep_neutral")
    rect(d, (9, 4, 9, 4), "deep_neutral")
    rect(d, (7, 4, 8, 4), "skin_detail")
    rect(d, (4, 6, 11, 12), "blanket_main")
    rect(d, (4, 12, 11, 13), "blanket_shadow")
    rect(d, (4, 6, 11, 7), "bed_linen")
    return image


def _down_arrow(d: ImageDraw.ImageDraw, colour: str, x: int = 8, top: int = 4) -> None:
    rect(d, (x - 1, top, x, top + 5), colour)
    polygon(d, [(x - 4, top + 5), (x + 3, top + 5), (x, top + 9)], colour)


def _up_arrow(d: ImageDraw.ImageDraw, colour: str, x: int = 8, bottom: int = 12) -> None:
    rect(d, (x - 1, bottom - 5, x, bottom), colour)
    polygon(d, [(x - 4, bottom - 5), (x + 3, bottom - 5), (x, bottom - 9)], colour)


def draw_bed_overlays() -> Image.Image:
    tiles: list[Image.Image] = []

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (10, 1, 14, 5), "white")
    rect(d, (11, 2, 13, 4), "ok_green")
    rect(d, (14, 3, 15, 3), "ok_green")
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (10, 1, 14, 5), "inactive_grey")
    rect(d, (11, 2, 13, 4), "white")
    line(d, [(10, 1), (14, 5)], "deep_neutral")
    rect(d, (12, 7, 12, 7), "inactive_grey")
    rect(d, (12, 9, 12, 9), "inactive_grey")
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    line(d, [(1, 1), (14, 1), (14, 14), (1, 14), (1, 1)], "alarm_amber")
    _down_arrow(d, "alarm_amber", top=3)
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (0, 0, 15, 2), "alarm_red")
    rect(d, (0, 13, 15, 15), "alarm_red")
    rect(d, (0, 3, 2, 12), "alarm_red")
    rect(d, (13, 3, 15, 12), "alarm_red")
    _down_arrow(d, "alarm_red", top=2)
    _down_arrow(d, "alarm_red", top=6)
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    line(d, [(1, 14), (1, 1), (14, 1), (14, 14), (1, 14)], "alarm_yellow")
    _up_arrow(d, "alarm_yellow", bottom=12)
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    polygon(d, [(1, 1), (14, 1), (8, 8)], "alarm_amber")
    _down_arrow(d, "white", top=1)
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    polygon(d, [(1, 14), (14, 14), (8, 7)], "alarm_yellow")
    _up_arrow(d, "deep_neutral", bottom=14)
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (2, 3, 9, 12), "deep_neutral")
    rect(d, (3, 4, 8, 8), "screen_glow")
    rect(d, (5, 10, 6, 11), "white")
    polygon(d, [(12, 4), (10, 8), (12, 11), (14, 8)], "alarm_red")
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    line(d, [(3, 8), (6, 11), (13, 3)], "ok_green", 2)
    rect(d, (2, 7, 4, 9), "white")
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    rect(d, (3, 5, 12, 13), "deep_neutral")
    rect(d, (4, 6, 11, 12), "purple")
    rect(d, (6, 3, 9, 5), "deep_neutral")
    rect(d, (7, 4, 8, 5), "white")
    line(d, [(4, 9), (11, 9)], "lilac")
    tiles.append(tile)

    tile = canvas((TILE, TILE))
    d = ImageDraw.Draw(tile)
    for offset in (0, 1):
        line(d, [(offset, 4), (offset, offset), (4, offset)], "screen_glow")
        line(d, [(11, offset), (15 - offset, offset), (15 - offset, 4)], "screen_glow")
        line(d, [(offset, 11), (offset, 15 - offset), (4, 15 - offset)], "screen_glow")
        line(d, [(11, 15 - offset), (15 - offset, 15 - offset), (15 - offset, 11)], "screen_glow")
    tiles.append(tile)

    assert len(tiles) == len(BED_OVERLAYS)
    sheet = canvas((len(tiles) * TILE, TILE))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * TILE, 0))
    return sheet


def draw_effect_overlays() -> Image.Image:
    # Mixed-size packed sheet: required 16x8 ring, then optional 16x16 busy icon.
    sheet = canvas((16, 24))
    d = ImageDraw.Draw(sheet)
    line(d, [(2, 3), (4, 1), (11, 1), (13, 3), (11, 6), (4, 6), (2, 3)], "white")
    line(d, [(1, 3), (4, 0), (11, 0), (14, 3), (11, 7), (4, 7), (1, 3)], "screen_glow")
    y = 8
    rect(d, (6, y + 2, 9, y + 4), "inactive_grey")
    rect(d, (5, y + 4, 10, y + 5), "deep_neutral")
    polygon(d, [(5, y + 6), (10, y + 6), (8, y + 9), (7, y + 9)], "inactive_grey")
    polygon(d, [(7, y + 10), (8, y + 10), (10, y + 13), (5, y + 13)], "inactive_grey")
    rect(d, (5, y + 14, 10, y + 15), "deep_neutral")
    return sheet


def recolour(image: Image.Image, substitutions: dict[int, str]) -> Image.Image:
    result = image.copy()
    palette = result.getpalette()
    assert palette is not None
    for index, colour in substitutions.items():
        r, g, b = _rgb(colour)
        palette[index * 3 : index * 3 + 3] = [r, g, b]
    result.putpalette(palette)
    result.info["transparency"] = 0
    return result


def draw_qa(
    tiles: Image.Image,
    characters: Image.Image,
    patient: Image.Image,
    bed_overlays: Image.Image,
    effects: Image.Image,
) -> None:
    # Large no-interpolation board for human visual inspection.
    scale = 8
    board = Image.new("RGBA", (896, 1056), _rgb("#E8E2D4") + (255,))
    tiles_rgba = tiles.convert("RGBA").resize((96 * scale, 96 * scale), Image.Resampling.NEAREST)
    board.alpha_composite(tiles_rgba, (64, 0))

    # Seven down-facing idle sprites at native scale and 8x.
    lineup = canvas((7 * CHAR_W, CHAR_H))
    for role_index in range(7):
        row = role_index * 4
        sprite = characters.crop((CHAR_W, row * CHAR_H, 2 * CHAR_W, (row + 1) * CHAR_H))
        lineup.paste(sprite, (role_index * CHAR_W, 0))
    board.alpha_composite(
        lineup.convert("RGBA").resize((7 * CHAR_W * scale, CHAR_H * scale), Image.Resampling.NEAREST),
        (0, 784),
    )

    overlay_strip = bed_overlays.convert("RGBA").resize(
        (bed_overlays.width * 4, bed_overlays.height * 4), Image.Resampling.NEAREST
    )
    board.alpha_composite(overlay_strip, (0, 976))
    effect_large = effects.convert("RGBA").resize(
        (effects.width * 4, effects.height * 4), Image.Resampling.NEAREST
    )
    board.alpha_composite(effect_large, (832, 952))
    board.save(QA / "asset_contact_sheet.png")

    # Every direction and animation frame, grouped role-by-role.
    frame_scale = 4
    gap = 8
    block_w = 3 * CHAR_W * frame_scale
    block_h = 4 * CHAR_H * frame_scale
    all_frames = Image.new(
        "RGBA",
        (len(CHARACTERS) * block_w + (len(CHARACTERS) - 1) * gap, block_h),
        _rgb("#E8E2D4") + (255,),
    )
    for role_index in range(len(CHARACTERS)):
        role_sheet = characters.crop(
            (0, role_index * 4 * CHAR_H, 3 * CHAR_W, (role_index + 1) * 4 * CHAR_H)
        )
        role_sheet = role_sheet.convert("RGBA").resize(
            (block_w, block_h), Image.Resampling.NEAREST
        )
        all_frames.alpha_composite(role_sheet, (role_index * (block_w + gap), 0))
    all_frames.save(QA / "characters_all_frames.png")

    # Silhouette test: native 1x row plus a nearest-neighbour 12x view.
    silhouette = Image.new("RGBA", lineup.size, (0, 0, 0, 0))
    alpha = lineup.convert("RGBA").getchannel("A")
    silhouette.paste((0, 0, 0, 255), mask=alpha)
    sil_board = Image.new("RGBA", (7 * CHAR_W * 12, CHAR_H * 12 + 40), (255, 255, 255, 255))
    sil_board.alpha_composite(silhouette, (0, 8))
    sil_board.alpha_composite(
        silhouette.resize((7 * CHAR_W * 12, CHAR_H * 12), Image.Resampling.NEAREST),
        (0, 40),
    )
    sil_board.save(QA / "silhouette_test.png")

    # Bed identity matrix proves the two palette regions can be swapped independently.
    bed_tile = tiles.crop((2 * TILE, 2 * TILE, 3 * TILE, 3 * TILE))
    identity = Image.new("RGBA", (8 * TILE * 6, 5 * TILE * 6), (232, 226, 212, 255))
    for skin_index, skin in enumerate(SKIN_TONES):
        for blanket_index, blanket in enumerate(BLANKET_COLOURS):
            variant = recolour(
                patient,
                {
                    IDX["skin_main"]: skin["main"],
                    IDX["skin_detail"]: skin["detail"],
                    IDX["skin_shadow"]: skin["shadow"],
                    IDX["blanket_main"]: blanket["main"],
                    IDX["blanket_shadow"]: blanket["shadow"],
                },
            )
            composite = bed_tile.convert("RGBA")
            composite.alpha_composite(variant.convert("RGBA"))
            composite = composite.resize((TILE * 6, TILE * 6), Image.Resampling.NEAREST)
            identity.alpha_composite(composite, (blanket_index * TILE * 6, skin_index * TILE * 6))
    identity.save(QA / "patient_identity_matrix.png")

    # 25x21 orthogonal context mock: four eight-bed bays, centre station,
    # right-side drug room and bottom entrance. This is visual QA, not map data.
    map_w, map_h = 25, 21
    ward = Image.new("RGBA", (map_w * TILE, map_h * TILE), (232, 226, 212, 255))

    def tile_at(one_based_index: int) -> Image.Image:
        index = one_based_index - 1
        return tiles.crop(
            (
                (index % 6) * TILE,
                (index // 6) * TILE,
                (index % 6 + 1) * TILE,
                (index // 6 + 1) * TILE,
            )
        ).convert("RGBA")

    for y in range(map_h):
        for x in range(map_w):
            floor_index = 1 if (x + y) % 2 == 0 else 2
            if 9 <= x <= 15 or 9 <= y <= 11:
                floor_index = 4
            ward.alpha_composite(tile_at(floor_index), (x * TILE, y * TILE))

    for x in range(map_w):
        ward.alpha_composite(tile_at(7), (x * TILE, 0))
    for y in range(1, map_h):
        ward.alpha_composite(tile_at(9), (0, y * TILE))
        ward.alpha_composite(tile_at(10), ((map_w - 1) * TILE, y * TILE))

    bed_positions = [
        (x, y)
        for x in (2, 6, 18, 22)
        for y in (2, 4, 6, 8, 12, 14, 16, 18)
    ]
    assert len(bed_positions) == 32
    for patient_index, (x, y) in enumerate(bed_positions):
        ward.alpha_composite(tile_at(15), (x * TILE, y * TILE))
        skin = SKIN_TONES[patient_index % len(SKIN_TONES)]
        blanket = BLANKET_COLOURS[patient_index % len(BLANKET_COLOURS)]
        variant = recolour(
            patient,
            {
                IDX["skin_main"]: skin["main"],
                IDX["skin_detail"]: skin["detail"],
                IDX["skin_shadow"]: skin["shadow"],
                IDX["blanket_main"]: blanket["main"],
                IDX["blanket_shadow"]: blanket["shadow"],
            },
        ).convert("RGBA")
        ward.alpha_composite(variant, (x * TILE, y * TILE))
        if patient_index in (3, 14, 25):
            overlay_index = {3: 2, 14: 3, 25: 8}[patient_index]
            overlay = bed_overlays.crop(
                (overlay_index * TILE, 0, (overlay_index + 1) * TILE, TILE)
            ).convert("RGBA")
            ward.alpha_composite(overlay, (x * TILE, y * TILE))

    station = [25, 26, 27, 28, 25, 30, 29, 27]
    for i, tile_index in enumerate(station):
        x = 10 + i % 4
        y = 9 + i // 4
        ward.alpha_composite(tile_at(tile_index), (x * TILE, y * TILE))

    ward.alpha_composite(tile_at(31), (24 * TILE, 7 * TILE))
    ward.alpha_composite(tile_at(34), (12 * TILE, 20 * TILE))

    staff_positions = [(12, 12), (10, 7), (14, 6), (8, 10), (16, 10), (12, 5), (12, 18)]
    for role_index, (x, y) in enumerate(staff_positions):
        row = role_index * 4
        sprite = characters.crop(
            (CHAR_W, row * CHAR_H, 2 * CHAR_W, (row + 1) * CHAR_H)
        ).convert("RGBA")
        ward.alpha_composite(sprite, (x * TILE, (y + 1) * TILE - CHAR_H))
    ring = effects.crop((0, 0, 16, 8)).convert("RGBA")
    ward.alpha_composite(ring, (12 * TILE, 13 * TILE - 8))

    ward = ward.resize((map_w * TILE * 2, map_h * TILE * 2), Image.Resampling.NEAREST)
    ward.save(QA / "ward_composition.png")


def build_manifest() -> dict:
    palette_entries = [
        {"index": index, "name": name, "hex": colour, "alpha": 0 if index == 0 else 255}
        for index, (name, colour) in enumerate(PALETTE)
    ]
    return {
        "schema_version": 1,
        "projection": "orthogonal_top_down",
        "licence": "CC0-1.0",
        "tile_size": [16, 16],
        "character_sprite_size": [16, 24],
        "display_scale": 2,
        "palette": {
            "shared_colour_count_including_transparent": len(PALETTE),
            "transparent_index": 0,
            "entries": palette_entries,
            "skin_indices": {
                "main": IDX["skin_main"],
                "detail": IDX["skin_detail"],
                "shadow": IDX["skin_shadow"],
                "variants": SKIN_TONES,
            },
            "blanket_indices": {
                "main": IDX["blanket_main"],
                "shadow": IDX["blanket_shadow"],
                "variants": BLANKET_COLOURS,
                "also_used_for": "patient_walking gown trim",
            },
        },
        "sheets": {
            "tiles.png": {
                "size": [96, 96],
                "layout": {"columns": 6, "rows": 6, "sprite_size": [16, 16]},
                "items": [
                    {
                        "index": i + 1,
                        "name": name,
                        "x": (i % 6) * 16,
                        "y": (i // 6) * 16,
                        "width": 16,
                        "height": 16,
                    }
                    for i, name in enumerate(TILE_NAMES)
                ],
            },
            "characters.png": {
                "size": [48, 672],
                "layout": {
                    "columns": 3,
                    "rows": 28,
                    "sprite_size": [16, 24],
                    "row_order": "character-major, then down/left/right/up",
                    "column_order": FRAMES,
                },
                "items": [
                    {
                        "character": role,
                        "direction": direction,
                        "frame": frame,
                        "x": frame_index * 16,
                        "y": (role_index * 4 + direction_index) * 24,
                        "width": 16,
                        "height": 24,
                    }
                    for role_index, role in enumerate(CHARACTERS)
                    for direction_index, direction in enumerate(DIRECTIONS)
                    for frame_index, frame in enumerate(FRAMES)
                ],
            },
            "patients_in_bed.png": {
                "size": [16, 16],
                "layout": {"columns": 1, "rows": 1, "sprite_size": [16, 16]},
                "items": [
                    {
                        "index": 1,
                        "name": "patient_in_bed_overlay",
                        "x": 0,
                        "y": 0,
                        "width": 16,
                        "height": 16,
                        "aligns_to": "tiles.png index 15",
                    }
                ],
            },
            "overlays_bed.png": {
                "size": [176, 16],
                "layout": {"columns": 11, "rows": 1, "sprite_size": [16, 16]},
                "items": [
                    {
                        "index": i + 1,
                        "name": name,
                        "x": i * 16,
                        "y": 0,
                        "width": 16,
                        "height": 16,
                    }
                    for i, name in enumerate(BED_OVERLAYS)
                ],
            },
            "overlays_effect.png": {
                "size": [16, 24],
                "layout": "mixed-size, packed top-to-bottom without padding",
                "items": [
                    {
                        "index": 1,
                        "name": "player_highlight_ring",
                        "x": 0,
                        "y": 0,
                        "width": 16,
                        "height": 8,
                        "anchor": "feet_bottom_centre",
                    },
                    {
                        "index": 2,
                        "name": "staff_busy_indicator",
                        "optional": True,
                        "x": 0,
                        "y": 8,
                        "width": 16,
                        "height": 16,
                        "anchor": "character",
                    },
                ],
            },
        },
    }


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# Ward CGM simulator pixel assets

This directory contains the orthogonal top-down pixel-art set described in
`docs/ASSET_BRIEF.md`.

- `tiles.png`: 36 ordered 16×16 tiles, 6×6 with no padding.
- `characters.png`: 84 ordered 16×24 sprites, 28 rows × 3 columns.
- `patients_in_bed.png`: one indexed 16×16 bed-aligned patient overlay.
- `overlays_bed.png`: 11 ordered 16×16 status overlays.
- `overlays_effect.png`: one required 16×8 player ring followed by the optional
  16×16 busy marker, packed vertically without padding.
- `assets-index.json`: coordinates, ordering, the shared palette, and swappable
  skin/blanket palette indices and tables.
- `docs/asset-qa/`: nearest-neighbour visual checks; not runtime assets.

All runtime sheets share a 48-entry indexed palette. Palette index 0 is fully
transparent. Outer silhouettes contain no partial alpha. Skin uses indices 28,
46 and 47; blanket/identity trim uses indices 36 and 37.

## Originality and licence

Every pixel here is drawn by `scripts/generate_ward_assets.py`, which is part of
this repository: the sheets are program output rather than imported artwork,
and re-running the script reproduces them byte for byte. They were written from
scratch against `docs/ASSET_BRIEF.md`. No Nintendo, Game Freak, Two Point,
Project Hospital, or other commercial-game artwork was traced, recoloured,
copied, or included.

The authors of this repository — Isabel Smith and Aatish Thakerar, per the root
`LICENSE` — therefore hold the rights in this artwork in full, and to the extent
permitted by law dedicate it to the public domain under **CC0 1.0 Universal**.
See `LICENSE-CC0.txt`.

Regenerate with:

```bash
python scripts/generate_ward_assets.py
```

Pillow is development-only; the simulator runtime remains pygame-ce-only.
""",
        encoding="utf-8",
    )
    (OUT / "LICENSE-CC0.txt").write_text(
        """CC0 1.0 Universal

The pixel-art assets in this directory (tiles.png, characters.png,
patients_in_bed.png, overlays_bed.png, overlays_effect.png, and the
accompanying assets-index.json) were created for this repository by its
authors, Isabel Smith and Aatish Thakerar, and are drawn in full by
scripts/generate_ward_assets.py, which is part of this repository. No
third-party artwork was traced, copied, recoloured or otherwise incorporated,
so the authors hold the relevant rights in their entirety.

To the extent possible under law, Isabel Smith and Aatish Thakerar have waived
all copyright and related or neighboring rights to these assets and dedicate
them to the public domain under CC0 1.0 Universal.

Full legal text: https://creativecommons.org/publicdomain/zero/1.0/legalcode
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)

    tiles = draw_tiles()
    characters = draw_characters()
    patient = draw_patient_in_bed()
    bed_overlays = draw_bed_overlays()
    effects = draw_effect_overlays()

    save_indexed(tiles, OUT / "tiles.png")
    save_indexed(characters, OUT / "characters.png")
    save_indexed(patient, OUT / "patients_in_bed.png")
    save_indexed(bed_overlays, OUT / "overlays_bed.png")
    save_indexed(effects, OUT / "overlays_effect.png")

    manifest = build_manifest()
    (OUT / "assets-index.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme()
    draw_qa(tiles, characters, patient, bed_overlays, effects)


if __name__ == "__main__":
    main()
