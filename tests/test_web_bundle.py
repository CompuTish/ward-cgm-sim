"""Import safety for the WebAssembly bundle.

The browser build ships only what pygbag can run: the standard library plus
pygame. If anything reachable from the web entrypoint imports numpy, gymnasium
or stable-baselines3, the demo dies with an ImportError in the browser - and
nothing in a normal local test run would reveal it, because those packages are
installed here.

So the check runs in a subprocess with those modules poisoned.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Modules that exist natively but are absent from the pygbag runtime.
FORBIDDEN = ("numpy", "gymnasium", "stable_baselines3", "torch")

# Everything the web build vendors. ward_cgm_sim.env is deliberately absent.
WEB_SAFE_MODULES = [
    "ward_cgm_sim.config",
    "ward_cgm_sim.core",
    "ward_cgm_sim.core.actions",
    "ward_cgm_sim.core.alarms",
    "ward_cgm_sim.core.bedflow",
    "ward_cgm_sim.core.eligibility",
    "ward_cgm_sim.core.engine",
    "ward_cgm_sim.core.glucose",
    "ward_cgm_sim.core.observations",
    "ward_cgm_sim.core.patient",
    "ward_cgm_sim.core.rewards",
    "ward_cgm_sim.core.staff",
    "ward_cgm_sim.core.ward_map",
    "ward_cgm_sim.agents",
    "ward_cgm_sim.agents.random_agent",
    "ward_cgm_sim.agents.rule_based",
    # render/ ships to the browser too. pygame_renderer imports numpy lazily
    # inside to_rgb_array(), which never runs there, so importing the module
    # itself has to stay clean.
    "ward_cgm_sim.render",
    "ward_cgm_sim.render.sprites",
    "ward_cgm_sim.render.pygame_renderer",
]

# The block happens when the module is *executed*, not when it is looked up.
# pygame asks `find_spec("numpy")` at import time purely to discover whether
# numpy exists, and refusing the lookup breaks pygame itself - which would fail
# the whole render layer for a question nobody asked. Only a real `import`
# reaches the loader, so this still catches exactly what it is meant to.
_GUARD = """
import sys
from importlib.machinery import ModuleSpec

_FORBIDDEN = {forbidden!r}

class _Loader:
    def __init__(self, name):
        self.name = name
    def create_module(self, spec):
        raise ImportError(
            "web bundle imported '" + self.name + "', which is not available in "
            "the pygbag runtime"
        )
    def exec_module(self, module):
        pass

class _Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in _FORBIDDEN:
            return ModuleSpec(name, _Loader(name))
        return None

sys.meta_path.insert(0, _Blocked())
"""


def _run_guarded(body: str) -> subprocess.CompletedProcess:
    script = _GUARD.format(forbidden=set(FORBIDDEN)) + body
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_guard_itself_works():
    """Positive control.

    Without this, a broken guard would let every case below pass vacuously -
    the whole suite would be asserting nothing at all.
    """
    result = _run_guarded("import numpy\n")
    assert result.returncode != 0, "guard failed to block a forbidden import"
    assert "not available in the pygbag runtime" in result.stderr


@pytest.mark.parametrize("module", WEB_SAFE_MODULES)
def test_web_safe_module_imports_without_native_dependencies(module):
    result = _run_guarded(f"import {module}\n")
    assert result.returncode == 0, (
        f"{module} cannot be imported in the browser runtime:\n{result.stderr}"
    )


def test_a_full_episode_runs_without_native_dependencies():
    """The demo does not merely import - it has to simulate."""
    body = """
from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.agents.rule_based import RuleBasedAgent

engine = WardEngine(SimConfig(), seed=1)
agent = RuleBasedAgent()
agent.reset()
steps = 0
while True:
    _o, _r, terminated, truncated, _i = engine.step(agent.act(engine))
    steps += 1
    if terminated or truncated:
        break
assert steps > 50, "episode ended suspiciously early: %d steps" % steps
print("OK", steps)
"""
    result = _run_guarded(body)
    assert result.returncode == 0, f"simulation failed in browser runtime:\n{result.stderr}"
    assert "OK" in result.stdout


def test_the_art_loads_and_draws_without_native_dependencies():
    """Importing the renderer is not enough - the sheets load on construction.

    `SpriteSheet()` reads five PNGs and a JSON manifest and does palette
    surgery on them. That is the code the browser actually runs, and it must
    manage it with the standard library plus pygame-ce alone, and without a
    display.
    """
    body = """
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
pygame.init()
from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.render.pygame_renderer import WardRenderer

engine = WardEngine(SimConfig(), seed=2)
renderer = WardRenderer(engine, headless=True)
assert renderer.sprites.using_assets, "the art did not load"
for _ in range(5):
    renderer.draw()
    engine.step(0)
seen = {renderer.surface.get_at((x, y))[:3]
        for x in range(0, renderer.width, 5)
        for y in range(0, renderer.height, 5)}
assert len(seen) > 20, "ward came out blank: %d colours" % len(seen)
print("OK", len(seen))
"""
    result = _run_guarded(body)
    assert result.returncode == 0, f"the art cannot load in the browser runtime:\n{result.stderr}"
    assert "OK" in result.stdout


def test_package_root_does_not_pull_in_gymnasium():
    """``import ward_cgm_sim`` must stay light.

    env.py subclasses gymnasium.Env; importing it eagerly from __init__ would
    break both the web build and any install without the training extras.
    """
    result = _run_guarded("import ward_cgm_sim\n")
    assert result.returncode == 0, result.stderr


def test_env_module_is_correctly_excluded_from_the_web_bundle():
    """Negative control, paired with a positive one.

    Asserting only that env.py fails to import would also pass if the module
    were missing or misspelled, so first confirm it imports fine natively.
    """
    native = subprocess.run(
        [sys.executable, "-c", "import ward_cgm_sim.env"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert native.returncode == 0, (
        f"positive control failed: env.py should import natively\n{native.stderr}"
    )

    guarded = _run_guarded("import ward_cgm_sim.env\n")
    assert guarded.returncode != 0, (
        "env.py imported under the browser guard; it must never be vendored"
    )


def test_no_web_safe_module_mentions_a_forbidden_import():
    """Static sweep, to catch a lazily-imported dependency the runtime checks
    above would miss."""
    package = REPO_ROOT / "ward_cgm_sim"
    offenders = []
    for path in package.rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        if path.name == "env.py":
            continue  # native-only by design
        text = path.read_text()
        for name in FORBIDDEN:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and name in stripped:
                    # The renderer imports numpy lazily inside rgb_array only,
                    # which never runs in the browser.
                    if "rgb_array" in text and path.name == "pygame_renderer.py":
                        continue
                    offenders.append(f"{relative}: {stripped}")
    assert not offenders, "forbidden imports in web-shipped modules:\n" + "\n".join(offenders)
