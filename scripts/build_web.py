#!/usr/bin/env python3
"""Build the WebAssembly demo with pygbag.

pygbag packages only the contents of the target directory, so `web/main.py`
cannot import `ward_cgm_sim` from the repo root. This script vendors the
web-safe part of the package into `web/`, verifies the vendored tree imports
without any native dependency, then runs pygbag.

`ward_cgm_sim/env.py` is deliberately NOT vendored: it subclasses gymnasium.Env
and pulls in numpy, neither of which exists in the pygbag runtime.

Usage:
    python scripts/build_web.py             # vendor app code, verify, build
    python scripts/build_web.py --serve     # ...then serve it locally
    python scripts/build_web.py --vendor-only

The runtime is fetched from the pygame-web CDN. `--vendor-runtime` will instead
download it and serve it same-origin, which removes the third-party code
dependency - but that path is UNFINISHED: the resulting build does not boot,
and its URL handling still assumes the 0.9.x `/archives/` layout rather than
the `/cdn/<version>/` layout 0.9.3 emits. Do not enable it without fixing both.
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
    command = [sys.executable, "-m", "pygbag", "--title", "Ward CGM simulator"]
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


PYGBAG_VERSION = "0.9"
CDN_BASE = f"https://pygame-web.github.io/archives/{PYGBAG_VERSION}/"
REPO_BASE = "https://pygame-web.github.io/archives/repo/"

# Everything the browser fetches at runtime. Vendoring these means the page
# executes no third-party code: without it, whoever controls that GitHub Pages
# site can run arbitrary JavaScript on the host domain.
RUNTIME_FILES = [
    "pythons.js",
    "browserfs.min.js",
    "empty.html",
    "empty.ogg",
    "vtx.js",
    "cpythonrc.py",
    "vt/xterm.js",
    "vt/xterm-addon-image.js",
    "vt/xterm.css",
    "cpython312/main.js",
    "cpython312/main.data",
    "cpython312/main.wasm",
]
REPO_FILES = ["index-090-cp312.json", "repodata.json"]


def _fetch(url: str, target: Path) -> int:
    import ssl
    import urllib.request

    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi is in the web extra
        context = ssl.create_default_context()

    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, context=context, timeout=120) as response:
        data = response.read()
    target.write_bytes(data)
    return len(data)


def vendor_runtime() -> None:
    """UNFINISHED. Download the pygbag runtime to serve it same-origin.

    Known broken in two ways, both of which must be fixed before this is used:
      1. The resulting build downloads everything and then fails to boot.
      2. CDN_BASE below assumes the `/archives/0.9/` layout; pygbag 0.9.3
         emits `/cdn/0.9.3/`, so the guard in this function will reject it.

    Kept because eliminating the third-party runtime is the right end state.

    The runtime is pinned to a version, so it is immutable and can be cached
    for a year - which is what keeps the bandwidth cost of self-hosting ~25 MB
    of WebAssembly down to something a personal site can absorb.
    """
    output = WEB_DIR / "build" / "web"
    runtime = output / "runtime"
    total = 0

    for name in RUNTIME_FILES:
        total += _fetch(CDN_BASE + name, runtime / name)
    for name in REPO_FILES:
        total += _fetch(REPO_BASE + name, runtime / "repo" / name)

    # Point the loader at the local copy instead of the CDN.
    index = output / "index.html"
    html = index.read_text()
    if CDN_BASE not in html:
        sys.exit("pygbag output did not contain the expected CDN base URL")
    html = html.replace(CDN_BASE, "./runtime/")
    html = html.replace("CDN URL : ./runtime/", "CDN URL : ./runtime/ (vendored)")
    index.write_text(html)

    # The package index is resolved separately, inside the runtime itself.
    rc = runtime / "cpythonrc.py"
    rc.write_text(rc.read_text().replace(REPO_BASE, "./runtime/repo/"))

    print(f"vendored runtime: {len(RUNTIME_FILES) + len(REPO_FILES)} files, "
          f"{total / 1024 / 1024:.1f} MB, no third-party fetches remain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="serve locally on :8000")
    parser.add_argument("--vendor-only", action="store_true")
    parser.add_argument(
        "--vendor-runtime",
        action="store_true",
        help="UNFINISHED: self-host the runtime instead of using the CDN. The "
             "resulting build does not currently boot; see module docstring.",
    )
    args = parser.parse_args()

    vendor()
    verify()
    if not args.vendor_only:
        build(serve=args.serve)
        if not args.serve and args.vendor_runtime:
            vendor_runtime()


if __name__ == "__main__":
    main()
