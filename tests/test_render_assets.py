"""Contract tests for the pixel-art sheets.

These assertions verify the durable renderer-facing contract: exact packing,
indexed recolour regions, hard alpha, shared palette limits, and distinct role
silhouettes. The QA images provide the complementary human squint test.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from PIL import Image


ASSETS = Path(__file__).parents[1] / "ward_cgm_sim" / "render" / "assets"


def pixels(image: Image.Image):
    """Return pixel values without Pillow 12's deprecated getdata alias."""
    getter = getattr(image, "get_flattened_data", None)
    return getter() if getter else image.getdata()


def load_manifest() -> dict:
    return json.loads((ASSETS / "assets-index.json").read_text(encoding="utf-8"))


def test_all_required_asset_sheets_exist_with_exact_dimensions():
    expected = {
        "tiles.png": (96, 96),
        "characters.png": (48, 672),
        "patients_in_bed.png": (16, 16),
        "overlays_bed.png": (176, 16),
        "overlays_effect.png": (16, 24),
    }
    assert len(expected) == 5, "positive control: the brief defines five sheets"
    for filename, size in expected.items():
        path = ASSETS / filename
        assert path.is_file(), filename
        with Image.open(path) as image:
            assert image.size == size, filename
            assert image.mode == "P", filename
            assert image.info.get("transparency") == 0, filename


def test_manifest_lists_every_packed_sprite_without_padding():
    manifest = load_manifest()
    sheets = manifest["sheets"]
    assert len(sheets["tiles.png"]["items"]) == 36
    assert len(sheets["characters.png"]["items"]) == 84
    assert len(sheets["overlays_bed.png"]["items"]) == 11
    assert len(sheets["overlays_effect.png"]["items"]) == 2

    tile_items = sheets["tiles.png"]["items"]
    assert tile_items[0]["name"] == "ward_floor_light"
    assert tile_items[14]["name"] == "hospital_bed_made"
    assert tile_items[-1]["name"] == "sluice_dirty_utility_door"
    assert {(item["x"], item["y"]) for item in tile_items} == {
        (x * 16, y * 16) for y in range(6) for x in range(6)
    }

    character_items = sheets["characters.png"]["items"]
    assert character_items[0]["character"] == "ward_nurse_player"
    assert character_items[0]["direction"] == "down"
    assert character_items[0]["frame"] == "left_step"
    assert character_items[-1]["character"] == "patient_walking"
    assert character_items[-1]["direction"] == "up"
    assert character_items[-1]["frame"] == "right_step"


def test_runtime_sheets_share_a_48_colour_palette_and_hard_alpha():
    manifest = load_manifest()
    palette = manifest["palette"]
    assert palette["shared_colour_count_including_transparent"] == 48
    assert len(palette["entries"]) == 48
    assert palette["transparent_index"] == 0

    filenames = list(manifest["sheets"])
    assert len(filenames) == 5, "positive control: inspect every runtime sheet"
    reference_palette = None
    for filename in filenames:
        with Image.open(ASSETS / filename) as image:
            current = image.getpalette()[: 48 * 3]
            reference_palette = current if reference_palette is None else reference_palette
            assert current == reference_palette, filename
            alpha_values = set(pixels(image.convert("RGBA").getchannel("A")))
            assert alpha_values
            assert alpha_values <= {0, 255}, filename


def test_skin_and_blanket_indices_are_present_and_independently_swappable():
    manifest = load_manifest()
    palette = manifest["palette"]
    skin = palette["skin_indices"]
    blanket = palette["blanket_indices"]
    assert len(skin["variants"]) == 5
    assert len(blanket["variants"]) == 8
    assert {skin["main"], skin["detail"], skin["shadow"]}.isdisjoint(
        {blanket["main"], blanket["shadow"]}
    )

    with Image.open(ASSETS / "patients_in_bed.png") as patient:
        indices = set(pixels(patient))
    assert {skin["main"], skin["detail"], skin["shadow"]} <= indices
    assert {blanket["main"], blanket["shadow"]} <= indices

    with Image.open(ASSETS / "characters.png") as characters:
        indices = set(pixels(characters))
    assert {skin["main"], skin["detail"], skin["shadow"]} <= indices
    assert blanket["main"] in indices, "walking-patient gown trim must preserve identity"


def test_every_character_frame_is_nonempty_and_bottom_anchored():
    with Image.open(ASSETS / "characters.png") as sheet:
        sprites = [
            sheet.crop((column * 16, row * 24, (column + 1) * 16, (row + 1) * 24))
            for row in range(28)
            for column in range(3)
        ]
    assert len(sprites) == 84
    for index, sprite in enumerate(sprites):
        alpha = sprite.convert("RGBA").getchannel("A")
        assert alpha.getbbox() is not None, index
        assert alpha.getbbox()[3] == 24, f"sprite {index} must touch the bottom anchor"


def test_front_idle_role_silhouettes_are_pairwise_distinct():
    with Image.open(ASSETS / "characters.png") as sheet:
        silhouettes = []
        for role_index in range(7):
            y = role_index * 4 * 24
            sprite = sheet.crop((16, y, 32, y + 24)).convert("RGBA")
            mask = tuple(value > 0 for value in pixels(sprite.getchannel("A")))
            silhouettes.append(mask)

    assert len(silhouettes) == 7, "positive control: all roles are represented"
    for (left_index, left), (right_index, right) in combinations(
        enumerate(silhouettes), 2
    ):
        assert left != right, f"roles {left_index} and {right_index} share a silhouette"
        intersection = sum(a and b for a, b in zip(left, right, strict=True))
        union = sum(a or b for a, b in zip(left, right, strict=True))
        assert union > 0
        assert intersection / union < 0.90, (
            f"roles {left_index} and {right_index} are too similar for the squint test"
        )


def test_reserved_clinical_colours_do_not_leak_into_identity_art():
    manifest = load_manifest()
    palette_by_name = {
        entry["name"]: entry["index"] for entry in manifest["palette"]["entries"]
    }
    reserved = {
        palette_by_name["alarm_red"],
        palette_by_name["alarm_amber"],
        palette_by_name["alarm_yellow"],
        palette_by_name["ok_green"],
    }

    with Image.open(ASSETS / "characters.png") as characters:
        assert set(pixels(characters)).isdisjoint(reserved)
    with Image.open(ASSETS / "patients_in_bed.png") as patient:
        assert set(pixels(patient)).isdisjoint(reserved)

    with Image.open(ASSETS / "tiles.png") as tiles:
        for tile_index in range(36):
            tile = tiles.crop(
                (
                    (tile_index % 6) * 16,
                    (tile_index // 6) * 16,
                    (tile_index % 6 + 1) * 16,
                    (tile_index // 6 + 1) * 16,
                )
            )
            used = set(pixels(tile)) & reserved
            if tile_index == 28:
                assert used == {palette_by_name["alarm_red"]}
            else:
                assert not used, f"reserved clinical colour leaked into tile {tile_index + 1}"


def test_up_facing_patients_show_the_open_backed_gown():
    """The gown's open back is a brief requirement and easy to lose silently.

    Without this, deleting the opening from the generator and regenerating
    would leave every other asset assertion green.
    """
    manifest = load_manifest()
    palette = {entry["name"]: entry["index"] for entry in manifest["palette"]["entries"]}
    opening = palette["deep_neutral"]

    items = manifest["sheets"]["characters.png"]["items"]
    facing = {
        direction: [i for i in items
                    if i["character"] == "patient_walking" and i["direction"] == direction]
        for direction in ("up", "down")
    }
    assert len(facing["up"]) == 3 and len(facing["down"]) == 3, "positive control"

    with Image.open(ASSETS / "characters.png") as sheet:
        def torso(item):
            # Torso band only: the head sits above y=13, so the dark eye pixels
            # on forward-facing frames cannot be mistaken for the opening.
            box = (item["x"], item["y"] + 13, item["x"] + 16, item["y"] + 20)
            return list(pixels(sheet.crop(box)))

        for item in facing["up"]:
            count = torso(item).count(opening)
            assert count > 0, f"no gown opening in the {item['frame']} up frame"
        for item in facing["down"]:
            assert torso(item).count(opening) == 0, (
                "the opening must only show from behind"
            )
