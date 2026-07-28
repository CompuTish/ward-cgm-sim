"""Tests for the renderer and the artwork it draws with.

The renderer had no coverage at all until the commissioned sheets were wired
in, which meant a full suite could pass while the demo drew nothing. These
cover the two things that can silently go wrong: the art failing to load (and
degrading to rectangles unnoticed), and the palette swap failing (leaving every
patient looking identical, which defeats the point of per-patient identity).

They also pin the POMDP boundary at the render layer: the map must not show the
viewer a clinical fact the policy has to spend a step learning.
"""

from __future__ import annotations

import os
import subprocess
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


def test_a_full_shift_renders_with_the_commissioned_art():
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


def test_a_patient_keeps_the_same_appearance_all_shift():
    sheet = SpriteSheet()
    first = sheet.patient_in_bed(11, 11)
    assert first is sheet.patient_in_bed(11, 11)
    assert first is not sheet.patient_in_bed(12, 12)


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
    # The alarm state of the board is a second monitor tile, only reachable
    # when something is alarming, so check it explicitly rather than by walking.
    assert "desk_monitor_alarm" in available
