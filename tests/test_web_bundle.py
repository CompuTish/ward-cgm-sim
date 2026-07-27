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
]

_GUARD = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        root = name.split('.')[0]
        if root in {forbidden!r}:
            raise ImportError(
                "web bundle imported '" + name + "', which is not available in "
                "the pygbag runtime"
            )
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
