# PREFLIGHT - run before making changes

Short by design. It catches the environment problems that otherwise surface as
confusing failures halfway through a task. **Report the results** before you
start work.

Run everything from the repository root.

---

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python --version          # expect 3.10+
```

The system `python` has none of the dependencies. If commands fail with
`ModuleNotFoundError`, the venv is not active - prefix with `.venv/bin/` or
activate it.

**Expected:** Python 3.10 or later, install completes.

---

## 2. The suite

```bash
python -m pytest -q
```

**Expected:** all tests pass, none skipped. A skip here usually means a
seed-dependent test quietly stopped exercising its case - investigate rather
than accept it.

---

## 3. Gymnasium contract

```bash
python -c "from gymnasium.utils.env_checker import check_env; \
           from ward_cgm_sim.env import WardCGMTelemetryEnv; \
           check_env(WardCGMTelemetryEnv(), skip_render_check=True); print('OK')"
```

**Expected:** `OK`. A render-mode warning is normal.

---

## 4. The model still behaves

```bash
python scripts/run_baseline.py --episodes 10 --seed 0
```

**Expected, roughly:** 144-step shifts, ~8 discharges, a peak queue in low
single digits, and few or no serious adverse events. Wildly different numbers
mean something in `config.py` has moved - find out what before proceeding.

**Record the numbers with the seed.** A figure quoted without its invocation is
not reproducible.

---

## 5. The counterfactual invariant

```bash
python -m pytest tests/test_counterfactual_rng.py -q
```

**Expected:** all pass. If you are touching anything involving `random`, run
this again afterwards - this is the guard on the entire comparison.

---

## 6. Web import safety (only if touching `core/`, `render/`, `agents/` or `config.py`)

```bash
python -m pytest tests/test_web_bundle.py -q
```

**Expected:** all pass. A failure means something in the browser-shipped tree
imports numpy, gymnasium or another native dependency, and the demo would break
once deployed.

---

## 7. Report

State plainly:

- Python version, whether install succeeded
- Test count and result
- The baseline numbers **and the seed**
- Anything skipped, and why
- Anything unexpected, even if it seems unrelated

If a step fails, stop and report it. Do not work around a failing preflight -
it is almost always telling you something true.
