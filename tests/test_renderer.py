"""Tests for the renderer and the artwork it draws with.

The renderer had no coverage at all until the art sheets were wired
in, which meant a full suite could pass while the demo drew nothing. These
cover the two things that can silently go wrong: the art failing to load (and
degrading to rectangles unnoticed), and the palette swap failing (leaving every
patient looking identical, which defeats the point of per-patient identity).

They also pin the POMDP boundary at the render layer: the map must not show the
viewer a clinical fact the policy has to spend a step learning.
"""

from __future__ import annotations

import json
import os
import subprocess
import types
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from ward_cgm_sim.agents.rule_based import RuleBasedAgent  # noqa: E402
from ward_cgm_sim.config import SimConfig  # noqa: E402
from ward_cgm_sim.core.alarms import AlarmKind  # noqa: E402
from ward_cgm_sim.core.engine import WardEngine  # noqa: E402
from ward_cgm_sim.core.patient import DischargeStage, Location  # noqa: E402
from ward_cgm_sim.render import pygame_renderer  # noqa: E402
from ward_cgm_sim.render.pygame_renderer import ALARM_OVERLAYS, WardRenderer  # noqa: E402
from ward_cgm_sim.render.sprites import SCALE, SOURCE_TILE, TILE, SpriteSheet  # noqa: E402

REPO_ROOT = Path(__file__).parents[1]
NO_ASSETS = Path("/nonexistent-ward-cgm-assets")


@pytest.fixture(scope="module", autouse=True)
def _display():
    pygame.init()
    pygame.display.set_mode((8, 8))
    yield
    pygame.display.quit()


def run_shift(steps: int = 40, assets_dir=None) -> WardRenderer:
    engine = WardEngine(SimConfig(), seed=7)
    renderer = WardRenderer(engine, headless=True, assets_dir=assets_dir)
    agent = RuleBasedAgent(engine.cfg)
    agent.reset()
    for _ in range(steps):
        renderer.draw()
        engine.step(agent.act(engine))
    renderer.draw()
    return renderer


def colours(surface: pygame.Surface, rect: pygame.Rect) -> set:
    return {
        surface.get_at((x, y))[:3]
        for x in range(rect.left, rect.right)
        for y in range(rect.top, rect.bottom)
    }


def test_a_full_shift_renders_with_the_real_art():
    renderer = run_shift()
    assert renderer.sprites.using_assets, "the sheets in render/assets should be used"

    ward = pygame.Rect(0, 0, renderer.engine.ward_map.width * TILE,
                       renderer.engine.ward_map.height * TILE)
    drawn = colours(renderer.surface, ward)
    # Rectangle art musters barely a dozen colours over the map; the sheets
    # share a 48-colour palette, so anything near the low end means the art
    # silently failed to load and the fallback took over.
    assert len(drawn) > 20, f"map drew only {len(drawn)} colours - is the art loading?"


def test_the_procedural_fallback_still_draws_a_whole_ward():
    renderer = run_shift(assets_dir=NO_ASSETS)
    assert renderer.sprites.using_assets is False
    assert renderer.sprites.character_y_offset == 0
    ward = pygame.Rect(0, 0, 200, 200)
    assert len(colours(renderer.surface, ward)) > 1, "fallback drew a blank ward"


def test_missing_art_falls_back_but_corrupt_art_is_not_hidden(tmp_path):
    assert SpriteSheet(NO_ASSETS).using_assets is False
    (tmp_path / "assets-index.json").write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(Exception):
        SpriteSheet(tmp_path)


def test_every_patient_identity_is_visibly_distinct():
    """The palette swap must actually change pixels.

    If `_recolour` silently no-ops, all forty combinations come back identical
    and every patient on the ward looks the same - the bug this test exists for.
    """
    sheet = SpriteSheet()
    assert sheet.using_assets
    assert sheet.n_skins == 5 and sheet.n_blankets == 8

    rendered = {
        pygame.image.tobytes(sheet.patient_in_bed(skin, blanket), "RGBA")
        for skin in range(sheet.n_skins)
        for blanket in range(sheet.n_blankets)
    }
    assert len(rendered) == sheet.n_skins * sheet.n_blankets

    faces = {
        pygame.image.tobytes(sheet.person("nurse", "down", 1, skin), "RGBA")
        for skin in range(sheet.n_skins)
    }
    assert len(faces) == sheet.n_skins, "skin tones are not being swapped"


def test_all_three_recolour_routes_agree(monkeypatch):
    """The browser runtime decides which route runs, so all three must match.

    `_recolour` repaints an indexed palette entry when the loaded surface is
    8-bit, matches on the baked colour via PixelArray when it is not, and falls
    back to a per-pixel loop if PixelArray is unavailable. Only the first is
    exercised on this machine, so the other two are pinned here.
    """
    from ward_cgm_sim.render import sprites

    indexed = pygame.image.load(str(sprites.ASSETS_DIR / "patients_in_bed.png"))
    assert indexed.get_bitsize() == 8, "positive control: the sheet is indexed"
    truecolour = indexed.convert_alpha()
    assert truecolour.get_bitsize() != 8

    # Repaint the baked dusty-blue blanket to the sage variant.
    swaps = [(36, "#7A98BA", "#8CA68C"), (37, "#657F9E", "#718A71")]

    via_palette = pygame.image.tobytes(sprites._recolour(indexed, swaps), "RGBA")
    via_pixelarray = pygame.image.tobytes(sprites._recolour(truecolour, swaps), "RGBA")

    monkeypatch.delattr(pygame, "PixelArray")
    via_loop = pygame.image.tobytes(sprites._recolour(truecolour, swaps), "RGBA")

    baseline = pygame.image.tobytes(truecolour, "RGBA")
    assert via_palette != baseline, "positive control: the swap must change pixels"
    assert via_palette == via_pixelarray
    assert via_palette == via_loop


def test_a_patient_keeps_the_same_appearance_all_shift():
    sheet = SpriteSheet()
    first = sheet.patient_in_bed(11, 11)
    assert first is sheet.patient_in_bed(11, 11)
    assert first is not sheet.patient_in_bed(12, 12)


def test_a_walking_patient_carries_their_bed_blanket_colour():
    """You must be able to follow one patient from their bed to the door.

    The gown trim is the same indexed region as the blanket, so a walking
    patient that ignores it comes out dusty blue for everyone - which is what
    happened before this was threaded through `person()`.
    """
    sheet = SpriteSheet()
    manifest = json.loads(
        (sheet.pack.directory / "assets-index.json").read_text(encoding="utf-8")
    )
    variants = manifest["palette"]["blanket_indices"]["variants"]
    assert len(variants) == 8, "positive control: eight blankets to tell apart"

    def trim_colours(blanket: int) -> set:
        sprite = sheet.person("patient", "down", 1, skin=0, blanket=blanket)
        return {
            sprite.get_at((x, y))[:3]
            for x in range(sprite.get_width())
            for y in range(sprite.get_height())
            if sprite.get_at((x, y))[3] > 0
        }

    for index, variant in enumerate(variants):
        expected = pygame.Color(variant["main"])[:3]
        assert expected in trim_colours(index), f"{variant['name']} trim missing"

    # And no two patients look alike.
    rendered = {
        pygame.image.tobytes(sheet.person("patient", "down", 1, 0, b), "RGBA")
        for b in range(len(variants))
    }
    assert len(rendered) == len(variants)

    # Staff are unaffected - a nurse has no blanket.
    nurse = pygame.image.tobytes(sheet.person("nurse", "down", 1, 0), "RGBA")
    assert nurse == pygame.image.tobytes(
        sheet.person("nurse", "down", 1, 0, blanket=3), "RGBA"
    )


def test_each_direction_and_walk_frame_is_a_distinct_sprite():
    sheet = SpriteSheet()
    facings = {
        pygame.image.tobytes(sheet.person("doctor", d, 1), "RGBA")
        for d in ("down", "left", "right", "up")
    }
    assert len(facings) == 4, "the four facings should not be the same picture"

    # Four-beat cycle: contact, pass, contact, pass - so phases 1 and 3 match.
    walk = [pygame.image.tobytes(sheet.person("doctor", "down", p), "RGBA")
            for p in range(4)]
    assert walk[1] == walk[3]
    assert len({tuple(w) for w in walk}) == 3


def test_characters_stand_on_the_tile_they_occupy():
    sheet = SpriteSheet()
    sprite = sheet.person("agent", "down", 1)
    assert sprite.get_height() == 24 * SCALE
    # Feet on the floor: the sprite bottom must land on the tile bottom.
    assert sheet.character_y_offset + sprite.get_height() == TILE
    assert sheet.character_y_offset == -(24 - SOURCE_TILE) * SCALE


def test_facing_follows_movement_and_is_remembered_when_still():
    renderer = run_shift(steps=1)
    key = "test-character"
    assert renderer._face(key, 1, 0) == "right"
    assert renderer._face(key, -1, 0) == "left"
    assert renderer._face(key, 0, 1) == "down"
    assert renderer._face(key, 0, -1) == "up"
    # Standing still keeps the last facing rather than snapping back to a default.
    assert renderer._face(key, 0, 0) == "up"
    # Diagonals resolve to the dominant axis.
    assert renderer._face(key, 5, -2) == "right"
    assert renderer._face(key, -2, 9) == "down"


def test_every_alarm_kind_can_be_drawn():
    sheet = SpriteSheet()
    assert set(ALARM_OVERLAYS) == set(AlarmKind), "an alarm kind has no overlay"
    for name in ALARM_OVERLAYS.values():
        assert sheet.bed_overlay(name) is not None, name


def test_the_map_never_reveals_discharge_readiness_the_agent_has_not_learned():
    """Rendering `discharge_stage` would leak hidden truth to the viewer.

    The engine knows a patient is ready long before the agent does; only
    `knowledge.known_discharge_ready` reflects what has actually been learned.
    """
    source = (REPO_ROOT / "ward_cgm_sim" / "render" / "pygame_renderer.py").read_text(
        encoding="utf-8"
    )
    assert "discharge_stage" not in source
    assert "known_discharge_ready" in source, "positive control: the learned fact is used"

    engine = WardEngine(SimConfig(), seed=3)
    renderer = WardRenderer(engine, headless=True)
    ready = [p for p in engine.flow.patients() if p.location is Location.BED]
    assert ready, "positive control: the ward must have someone in a bed"
    patient = ready[0]
    patient.discharge_stage = DischargeStage.READY
    patient.knowledge.known_discharge_ready = None

    x, y = engine.ward_map.bed_tile(patient.bed)
    tile = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
    renderer.draw()
    hidden = colours(renderer.surface, tile)

    patient.knowledge.known_discharge_ready = True
    renderer.draw()
    learned = colours(renderer.surface, tile)
    assert hidden != learned, "the overlay must appear once the fact is known"


def test_rgb_array_rendering_works_without_a_display():
    """The Gymnasium `rgb_array` path never calls `display.set_mode()`.

    `convert_alpha()` adopts the display format and raises without one, so
    loading the art that way broke `env.render()` in any clean process. Every
    other test in this file installs a video mode in a fixture, which hid it -
    hence the subprocess.
    """
    script = (
        "import os;os.environ['SDL_VIDEODRIVER']='dummy';"
        "from ward_cgm_sim.env import WardCGMTelemetryEnv;"
        "e=WardCGMTelemetryEnv(render_mode='rgb_array');e.reset(seed=1);"
        "f=e.render();"
        "assert f.shape[2]==3, f.shape;"
        "print('OK', f.shape[0], f.shape[1], len({tuple(p) for r in f[::7] for p in r[::7]}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    tag, height, width, distinct = result.stdout.strip().splitlines()[-1].split()
    assert tag == "OK"
    # Derived, not hard-coded: TILE is deliberately adjustable and a literal
    # here would just have to be chased every time it moves.
    ward_map = WardEngine(SimConfig(), seed=1).ward_map
    assert (int(height), int(width)) == (
        ward_map.height * TILE,
        ward_map.width * TILE + pygame_renderer.HUD_WIDTH,
    )
    # Positive control: a frame that raised, or drew nothing, is not a pass.
    assert int(distinct) > 20, "the frame came back nearly blank"


def test_the_render_is_identical_across_processes():
    """Staff drift must not depend on PYTHONHASHSEED.

    `hash()` of a str is salted per process, so using it to place sprites makes
    two runs of the same seed draw different frames.
    """
    script = (
        "import os;os.environ['SDL_VIDEODRIVER']='dummy';"
        "import hashlib,pygame;pygame.init();pygame.display.set_mode((8,8));"
        "from ward_cgm_sim.config import SimConfig;"
        "from ward_cgm_sim.core.engine import WardEngine;"
        "from ward_cgm_sim.render.pygame_renderer import WardRenderer;"
        "from ward_cgm_sim.agents.rule_based import RuleBasedAgent;"
        "e=WardEngine(SimConfig(),seed=5);r=WardRenderer(e,headless=True);"
        "a=RuleBasedAgent(e.cfg);a.reset();"
        "[(r.draw(),e.step(a.act(e))) for _ in range(30)];r.draw();"
        "print(hashlib.sha256(pygame.image.tobytes(r.surface,'RGBA')).hexdigest())"
    )
    digests = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        digests.append(result.stdout.strip().splitlines()[-1])
    assert digests[0] == digests[1], "the frame changed with the hash seed"


def test_the_web_bundle_ships_the_artwork():
    """pygbag packages a directory tree; art left behind means a blank demo."""
    names = {
        "tiles.png", "characters.png", "patients_in_bed.png",
        "overlays_bed.png", "overlays_effect.png", "assets-index.json",
    }
    assets = REPO_ROOT / "ward_cgm_sim" / "render" / "assets"
    assert names <= {p.name for p in assets.iterdir()}

    # The build output is gitignored, so only assert on it when it is there -
    # but when it is, it is the only thing that proves the art really shipped.
    apk = REPO_ROOT / "web" / "build" / "web" / "web.apk"
    if not apk.is_file():
        pytest.skip("no build output; run scripts/build_web.py first")

    import zipfile

    with zipfile.ZipFile(apk) as bundle:
        packaged = {Path(name).name for name in bundle.namelist()
                    if "render/assets/" in name}
    assert names <= packaged, f"missing from the bundle: {sorted(names - packaged)}"


def test_the_renderer_asks_for_tiles_that_exist():
    """Every tile name the renderer can emit must be in the sheet."""
    renderer = run_shift(steps=1)
    ward_map = renderer.engine.ward_map
    asked = {
        renderer._tile_name(ward_map.tile(x, y), x, y)
        for y in range(ward_map.height)
        for x in range(ward_map.width)
        if ward_map.tile(x, y) != pygame_renderer.BED
    }
    assert len(asked) > 5, "positive control: the ward uses a variety of tiles"
    available = set(renderer.sprites.pack.tiles)
    assert asked <= available, f"missing tiles: {sorted(asked - available)}"


def test_the_station_board_shows_both_monitor_states():
    """Walking a quiet ward only ever reaches the calm monitor tile."""
    renderer = run_shift(steps=1)
    station = renderer.engine.ward_map.station_tiles
    monitor = (min(x for x, _ in station) + 2, min(y for _, y in station))

    renderer.engine.active_alarms.clear()
    renderer.engine.cfg.telemetry_enabled = True
    assert renderer._alarm_showing() is False
    assert renderer._station_name(*monitor) == "desk_monitor_on"

    renderer.engine.active_alarms[0] = types.SimpleNamespace(
        kind=next(iter(AlarmKind)), resolved_step=None
    )
    assert renderer._alarm_showing() is True
    assert renderer._station_name(*monitor) == "desk_monitor_alarm"

    # Two genuinely different pictures, not the same tile named twice.
    tiles = renderer.sprites.pack.tiles
    assert pygame.image.tobytes(tiles["desk_monitor_on"], "RGBA") != pygame.image.tobytes(
        tiles["desk_monitor_alarm"], "RGBA"
    )

    # Telemetry off: the board stays calm even with an alarm object present,
    # because in that arm there is no dashboard to light up.
    renderer.engine.cfg.telemetry_enabled = False
    assert renderer._station_name(*monitor) == "desk_monitor_on"


def test_the_ward_floor_is_drawn_under_every_feature_tile():
    """Beds, desks and doors are drawn with transparent margins.

    With no floor beneath them the HUD's panel fill showed through and every
    bed gained a black surround - glaring on screen, and invisible to every
    other assertion in this file. PANEL_BG is not in the 48-colour art palette,
    so finding it anywhere over the map means a tile is missing its ground layer.
    """
    renderer = run_shift(steps=20)
    ward_map = renderer.engine.ward_map
    ward = pygame.Rect(0, 0, ward_map.width * TILE, ward_map.height * TILE)
    drawn = colours(renderer.surface, ward)
    assert len(drawn) > 20, "positive control: the map must actually be drawn"
    assert pygame_renderer.PANEL_BG not in drawn, (
        "the panel background is showing through a feature tile"
    )


def test_colleagues_walk_the_ward_rather_than_pacing_a_few_tiles():
    """They used to oscillate over three tiles beside a fixed anchor.

    That reads as a glitch, not a ward. Each colleague now walks to a random
    destination and on to the next, so over a shift they cover real ground.
    """
    renderer = run_shift(steps=1)
    visited = {role: {position} for role, position in renderer._staff_at.items()}
    assert len(visited) == 5, "positive control: five colleagues on the ward"

    for _ in range(600):
        renderer.draw()
        for role, position in renderer._staff_at.items():
            visited[role].add(position)

    for role, tiles in visited.items():
        assert len(tiles) > 10, f"{role} only reached {len(tiles)} tiles"


def test_colleagues_stay_on_walkable_ground_and_out_of_each_other():
    renderer = run_shift(steps=1)
    ward_map = renderer.engine.ward_map
    for _ in range(400):
        renderer.draw()
        positions = list(renderer._staff_at.values())
        for role, (x, y) in renderer._staff_at.items():
            assert ward_map.walkable(x, y), f"{role} walked into {(x, y)}"
        assert len(positions) == len(set(positions)), "two colleagues on one tile"


def test_colleague_movement_never_touches_the_simulation():
    """Presentation state only - CONTEXT_PACK section 4.

    If staff drew from `engine.rng`, drawing the ward would change the ward,
    and the telemetry-versus-routine comparison would be meaningless.
    """
    engine = WardEngine(SimConfig(), seed=4)
    renderer = WardRenderer(engine, headless=True)
    before_rng = engine.rng.getstate()
    before_state = (engine.step_index, engine.agent_x, engine.agent_y,
                    len(engine.flow.queue))
    patient_states = [
        (p.patient_id, p.true_glucose, p.rng.getstate(), p.rng_sensor.getstate())
        for p in engine.flow.patients()
    ]
    assert patient_states, "positive control: the ward must have patients"

    for _ in range(300):
        renderer.draw()

    assert engine.rng.getstate() == before_rng, "rendering consumed the ward RNG"
    assert (engine.step_index, engine.agent_x, engine.agent_y,
            len(engine.flow.queue)) == before_state
    after = [
        (p.patient_id, p.true_glucose, p.rng.getstate(), p.rng_sensor.getstate())
        for p in engine.flow.patients()
    ]
    assert after == patient_states, "rendering disturbed a patient stream"
