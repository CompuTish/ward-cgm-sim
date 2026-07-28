# TASK_START - read this first

Entrypoint for any AI agent or new contributor working on `ward-cgm-sim`.
Follow it in order. It should take a few minutes, not an hour.

## 1. Understand what this is

An **academic POMDP simulator** of a 32-bed hospital ward, built to explore
whether inpatient CGM telemetry improves ward workflow and patient safety. It
supports a Master's research project.

**It is not clinical software.** Every threshold, treatment pathway and timing
is a configurable placeholder with no external authority. If a change would make
the model look more like clinical decision support, that is a reason to reject
it, not a feature.

## 2. Read these, in this order

1. `docs/llm/CONTEXT_PACK.md` - the architecture, the invariants, and the
   traps. **This is the important one.** Most mistakes made on this repo are
   things it warns about.
2. `CLAUDE.md` - operating rules, import policy, testing expectations.
3. `docs/POMDP.md` - the formal specification, if you are touching the model
   itself. Skip for pure tooling work.

## 3. Complete the preflight

Work through `docs/llm/PREFLIGHT.md` and **report the results** before making
changes. It is short and it catches the environment problems that otherwise
surface as confusing failures halfway through a task.

## 4. Know the two rules that are easiest to break

Both have been broken before, both produced results that looked fine:

- **Counterfactual integrity.** Random draws are partitioned per patient so the
  telemetry and routine-monitoring arms simulate the same ward. A draw placed
  behind `if patient.is_enrolled`, or taken from a stream shared across
  patients, silently invalidates every comparison. See CONTEXT_PACK §4.
- **The web import boundary.** `ward_cgm_sim/env.py` imports gymnasium and
  numpy and must never reach the browser bundle. See CONTEXT_PACK §3.

## 5. Prove your tests bite

When you add a test, reintroduce the bug it targets, confirm the test fails,
then restore. Four tests in this repository's history passed against the very
defects they were written for. Assertions on internal state, and universal
claims over collections that might be empty, are the usual culprits.

## 6. Before you say you are done

- `python -m pytest` passes (from the repo root, with the venv active)
- If simulation behaviour changed: re-run `scripts/run_baseline.py` and
  `scripts/evaluate.py` and **state how the numbers moved, with the seed**
- If the web build changed: rebuild and load it in a real browser
- Say what you did not verify

## Common tasks

| Task | Start here |
|---|---|
| Change a clinical parameter | `ward_cgm_sim/config.py`, then re-run evaluate and report the delta |
| Add or change an action | `core/actions.py` → `core/engine.py::_resolve_action` → `docs/POMDP.md` §4 |
| Change the reward | `config.py::RewardConfig` → `core/rewards.py` → `docs/POMDP.md` §6 |
| Change what the agent can see | `core/observations.py` → tests in `tests/test_observability.py` |
| Change the art | `render/assets/assets-index.json` is the contract; load it in `render/sprites.py`, draw it in `render/pygame_renderer.py`. Read CONTEXT_PACK §6 first - **never** edit `web/ward_cgm_sim/`, which is generated |
| Regenerate the art | `scripts/generate_ward_assets.py` (dev-only, needs Pillow); output must stay byte-identical unless you meant to change it |
| Rebuild the web demo | `python scripts/build_web.py` (add `--serve` to preview the built output), then deploy per the parent repo's CLAUDE.md |
