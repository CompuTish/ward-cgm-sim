#!/usr/bin/env python3
"""Build the WebAssembly demo with pygbag.

pygbag packages only the contents of the target directory, so `web/main.py`
cannot import `ward_cgm_sim` from the repo root. This script vendors the
web-safe part of the package into `web/`, verifies the vendored tree imports
without any native dependency, then runs pygbag.

`ward_cgm_sim/env.py` is deliberately NOT vendored: it subclasses gymnasium.Env
and pulls in numpy, neither of which exists in the pygbag runtime.

Usage:
    python scripts/build_web.py           # vendor, verify, build
    python scripts/build_web.py --serve   # ...then serve it locally
    python scripts/build_web.py --vendor-only
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "ward_cgm_sim"
WEB_DIR = REPO_ROOT / "web"
VENDORED = WEB_DIR / "ward_cgm_sim"

# Everything the browser build needs. env.py is excluded on purpose.
VENDOR_INCLUDE = ["__init__.py", "config.py", "core", "render", "agents"]
EXCLUDE_NAMES = {"__pycache__", "env.py"}


def vendor() -> None:
    if VENDORED.exists():
        shutil.rmtree(VENDORED)
    VENDORED.mkdir(parents=True)

    for name in VENDOR_INCLUDE:
        source = PACKAGE / name
        target = VENDORED / name
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns(*EXCLUDE_NAMES)
            )
        else:
            shutil.copy2(source, target)

    stray = VENDORED / "env.py"
    if stray.exists():  # pragma: no cover - defensive
        stray.unlink()

    print(f"vendored {len(VENDOR_INCLUDE)} entries into {VENDORED.relative_to(REPO_ROOT)}")


def verify() -> None:
    """Fail the build if the vendored tree touches a native dependency."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_web_bundle.py", "-q"],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        sys.exit(
            "\nweb bundle import-safety check FAILED - the demo would break in "
            "the browser. Fix the offending import before building."
        )
    if (VENDORED / "env.py").exists():
        sys.exit("env.py was vendored into the web bundle; it must not be.")
    print("import-safety check passed")


def build(serve: bool) -> None:
    command = [sys.executable, "-m", "pygbag"]
    if not serve:
        command.append("--build")
    command.append(str(WEB_DIR))
    print("running:", " ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit("pygbag build failed")

    output = WEB_DIR / "build" / "web"
    if not serve and output.exists():
        print(f"\nbuild output: {output.relative_to(REPO_ROOT)}")
        for item in sorted(output.iterdir()):
            print(f"  {item.name}  ({item.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="serve locally on :8000")
    parser.add_argument("--vendor-only", action="store_true")
    args = parser.parse_args()

    vendor()
    verify()
    if not args.vendor_only:
        build(serve=args.serve)


if __name__ == "__main__":
    main()
