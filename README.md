# ward-cgm-sim

A runnable reinforcement-learning simulator of a hospital ward, built to explore
whether **inpatient continuous glucose monitoring (CGM) with telemetry** improves
ward workflow and patient safety.

You are the nurse. Thirty-two beds, a twelve-hour shift, five minutes a step. A
telemetry dashboard at the nurse station tells you when somebody's glucose is
heading the wrong way — sometimes. Sensors drift, alarm on nothing, and
occasionally just stop reporting without saying so. Meanwhile there are patients
to enrol, notes to check, consent to seek, discharges to chase and an ED queue
filling up behind you.

> ### Academic model — not clinical decision support
> This is a research and teaching model built for an MRes project. Every
> treatment pathway, threshold and timing is a **simplified, configurable,
> guideline-inspired placeholder**, chosen to make the simulation behave
> plausibly. Nothing here is prescribing guidance or bedside instruction, and it
> must not be used to inform real patient care. All clinical parameters live in
> [`ward_cgm_sim/config.py`](ward_cgm_sim/config.py) precisely so they can be
> varied for sensitivity analysis.

## What it models

- A 32-bed mixed medical/surgical ward, ED/admissions queue, transfers,
  discharges and unsafe overcrowding
- CGM telemetry: interstitial lag, per-sensor bias, noise, artefacts, false
  alarms, and **silent** signal loss that raises no alarm at all
- Point-of-care capillary glucose as the confirmatory reference, trusted over CGM
- Enrolment and de-enrolment against the study's inclusion/exclusion criteria,
  including mid-shift changes that make somebody ineligible
- Background nurses, HCAs, doctors, surgeons and the diabetes team, each
  available only some of the time — and only discoverable by asking
- A Pokémon-style top-down ward you actually walk around, because where you are
  standing determines what you can see and do

The formal specification — POMDP tuple, observation and action spaces,
transition dynamics, reward weights, and every clinical model — is in
**[`docs/POMDP.md`](docs/POMDP.md)**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/play.py                     # play a shift yourself
python scripts/play.py --watch             # watch the rule-based nurse
python scripts/run_baseline.py --episodes 10
python scripts/evaluate.py --episodes 30   # telemetry vs routine monitoring
python -m pytest
```

### Controls

Arrow keys or WASD to walk. Bedside actions apply to the bed you are standing
next to. Checking the dashboard works anywhere but costs a step, and what you
learn goes stale; standing at the nurse station refreshes it for free.

`D` dashboard · `C` check patient · `N` notes · `K` consent · `E` enrol ·
`R` review eligibility · `X` de-enrol · `SPACE` respond to alarm ·
`G` point-of-care glucose · `1` treat hypo · `2` treat hyper · `Q` escalate ·
`F1`–`F4` ask HCA/nurse/doctor/surgeon · `T` troubleshoot sensor ·
`P` support discharge · `B` prioritise bed flow · `TAB` hand over to the policy

## Using it as a Gymnasium environment

```python
from ward_cgm_sim.env import WardCGMTelemetryEnv

env = WardCGMTelemetryEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`Discrete(24)` actions, a `Box(296,)` observation with an explicit `-1`
*unknown* sentinel, and an `info["kpi"]` dictionary carrying every outcome
measure. `info["reward_components"]` breaks the return down by named component,
which matters far more than the scalar for interpreting a policy.

Training extras (`pip install -e ".[train]"`) add Stable-Baselines3;
`scripts/train_ppo.py` is a worked example.

## The experiment

The counterfactual is **telemetry versus routine monitoring**, not telemetry
versus nothing. Every patient still receives routine capillary rounds and
symptom recognition; CGM is additive. The question is whether the alarm gets
there first, and by how much.

```bash
python scripts/evaluate.py --episodes 30
```

Both arms run on **matched seeds with partitioned random streams**, so they
simulate the same ward — an intervention on one patient cannot perturb another's
trajectory. The primary outcome is hypoglycaemia **detection delay** within the
monitored cohort; secondary outcomes cover alarm burden, enrolment quality and
ward flow.

### What it currently shows

Rule-based policy, 60 matched shifts, `--seed 0`, default configuration.
Outcomes are pooled at the event level across shifts:

| Outcome (monitored cohort) | Telemetry | Routine monitoring |
|---|---|---|
| Episodes detected | 74% (23/31) | 33% (10/30) |
| Detection delay, given detected | 4.9 steps (~24 min) | 10.8 steps (~54 min) |

Read those two rows together, never separately. Delay is **conditional on
detection** and therefore censored — an episode nobody ever found contributes
no delay at all, so an arm that only catches the most obvious events looks
deceptively fast.

Ward-wide the same effect nearly vanishes, and that is the more interesting
result: only about one patient in seven meets the eligibility criteria, so a
large within-cohort improvement dilutes to very little at ward level. A real
study would need to be powered for the cohort, and would have to decide whether
a ward-level effect is the right thing to look for at all.

These are model outputs on a deliberately signal-enriched population, not
predictions, and **no parameter here is derived from primary data**. See
"Calibration and limitations" in `docs/POMDP.md` for what must be sourced
before this informs any real study design.

This is a simulation study of a workflow model. It cannot demonstrate clinical
benefit — it can show whether a mechanism plausibly produces one, and which
assumptions that conclusion depends on.

## Web demo

`python scripts/build_web.py` vendors the simulation core into `web/`, verifies
it imports without any native dependency, and builds a WebAssembly bundle with
pygbag. Add `--serve` to run it locally.

The core is **standard library plus pygame-ce only** so the same code runs
natively and in the browser. `ward_cgm_sim/env.py` (which needs gymnasium and
numpy) is deliberately excluded from the bundle;
`tests/test_web_bundle.py` enforces that in a subprocess with those packages
blocked.

## Layout

```
ward_cgm_sim/
  config.py       every tunable and reward weight
  core/           patient, glucose, alarms, eligibility, staff, bedflow,
                  ward_map, actions, observations, rewards, engine
  env.py          Gymnasium wrapper (native only)
  render/         procedural pixel sprites + top-down renderer
  agents/         random and rule-based baselines
scripts/          play, run_baseline, evaluate, train_ppo, build_web
tests/            130 tests; see CLAUDE.md for what each layer must prove
docs/POMDP.md     the formal specification
```

## Licence

MIT. Sprites are drawn procedurally in code — there are no third-party assets.
