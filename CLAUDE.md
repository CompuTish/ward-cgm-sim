# CLAUDE.md - ward-cgm-sim

An academic POMDP simulator of inpatient CGM telemetry on a 32-bed ward, built for an MRes
project. Gymnasium environment + pygame top-down renderer, compiled to WebAssembly with pygbag
and deployed to `ward-cgm-demo.web.app`.

**New here? Start at [`docs/llm/TASK_START.md`](docs/llm/TASK_START.md)**, which routes you
through `CONTEXT_PACK.md` (architecture, invariants, traps) and `PREFLIGHT.md` (checks to run
before changing anything). This file covers the rules; those cover the shape of the thing.

## Clinical-content rule (never violate)

This is a research model, not a clinical tool. Every treatment pathway, threshold and timing is a
simplified, configurable, guideline-inspired **placeholder**. Never present any of it as clinical
decision support, prescribing guidance, or bedside instruction - in code, docstrings, README,
commit messages, the web demo, or anything else user-facing. Where a clinical number appears,
state that it is configurable and where it lives (`ward_cgm_sim/config.py`).

## Operating principles

1. **Think before coding.** State assumptions, surface ambiguity, recommend rather than survey,
   and push back when a simpler path exists.
2. **Simplicity first.** Minimum code that solves the problem. No speculative features or
   unrequested configurability - with one deliberate exception: clinical parameters belong in
   `config.py` rather than inline, because sensitivity analysis is the point of the model.
3. **Surgical changes.** Touch only what the task requires; match local style.
4. **Goal-driven execution.** Name the verifiable success criterion before starting, then loop
   until it holds.

## Import policy (breaks the browser build if violated)

The core must stay importable with the standard library plus `pygame-ce` alone, because the same
tree is vendored into the WebAssembly bundle.

- `ward_cgm_sim/core/**`, `render/**`, `agents/**`, `config.py` - **stdlib + pygame-ce only**.
  No numpy, no gymnasium, no stable-baselines3.
- `ward_cgm_sim/env.py` - the only module allowed to import gymnasium/numpy. It must never be
  imported from `ward_cgm_sim/__init__.py`, and never vendored into `web/`.

## Testing

Match the layer to the change; if a layer doesn't apply, say why rather than skipping silently.
**If no applicable test exists, write one - an empty `tests/` directory is not coverage.**

Run from the repo root with the virtualenv active (`source .venv/bin/activate`), or prefix each
command with `.venv/bin/` - the system `python` has none of the dependencies.

```bash
python -m pytest                                   # unit + contract
python -c "from gymnasium.utils.env_checker import check_env; \
           from ward_cgm_sim.env import WardCGMTelemetryEnv; \
           check_env(WardCGMTelemetryEnv())"       # Gymnasium API contract
python scripts/run_baseline.py --episodes 10       # full shifts, end to end
python scripts/evaluate.py --episodes 30           # telemetry vs routine monitoring
```

What each layer must prove:

- **Unit** - deterministic logic: every branch of the eligibility and de-enrolment rules (each
  inclusion criterion, each exclusion, each mid-shift change), alarm thresholds including silent
  signal loss, reward accounting, and reproducibility under a fixed seed.
- **Integration** - a full 144-step shift runs without error and terminates for a legitimate
  reason; the Gymnasium contract passes `check_env`.
- **Behavioural** - when a change alters simulation dynamics, re-run the baseline and the
  evaluation and **state how the numbers moved**. A simulator that runs is not a simulator that
  behaves plausibly. Sanity-check outputs against what a real ward would do - incidence rates,
  length of stay, alarm burden, discharge throughput - and say so when they disagree.

## Verification discipline

1. **Verify durable state, not transient signals.** A test passing or a script exiting 0 proves
   an event, not an outcome. Name the post-condition and check *that*.
2. **"Done at this layer" is never "done."** Code written ≠ tests pass ≠ behaviour plausible ≠
   demo runs in a browser. Each "done" names the next unverified layer.
3. **Negative assertions need a positive control.** "No alarm fired" needs a companion assertion
   proving the simulation actually ran and the patient reached the state under test.
4. **Guard against vacuous truth.** Before asserting a property over a collection
   (`all(...)`, `len(x) == 0`), assert it is non-empty. An empty list satisfies every universal
   claim, so a broken producer reads as a pass - an easy way to false-green a whole KPI suite.
5. **Counterfactual integrity (a hard invariant, and easy to break).** The arms are comparable
   only because of common random numbers, partitioned by scope:

   - `engine.rng` - ward level: patient sampling, arrivals, staff. Must consume identically in
     both arms, so no draw on it may be conditional on telemetry or on an agent action.
   - `PatientState.rng` / `.rng_sensor` / `.rng_care` / `.rng_action` - that patient's
     physiology and flow, their CGM chain, routine checks on them, and the consequences of the
     agent acting on them. All seeded by domain-separated string seeds; never seed by XOR or
     multiply, which collides.

   **The invariant: an intervention on one patient may change how many draws *that* patient
   consumes, but must not change any other patient's random outcomes.** A draw gated behind
   `if patient.is_enrolled`, or taken from a stream shared across patients, breaks it - and the
   output still looks entirely plausible, which is what makes it dangerous.

   What this deliberately does *not* remove: beds and staff are genuinely shared, so treating
   one patient can still delay another through contention, and freeing a bed sooner really does
   admit the next patient sooner. Those are real ward effects. Only *spurious* coupling through
   randomness is eliminated.

   Any change touching a `random` call must re-run `tests/test_counterfactual_rng.py`, and any
   new test there must be mutation-checked: reintroduce the bug, confirm it fails, restore.
