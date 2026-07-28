# CONTEXT_PACK - architecture, invariants and traps

Everything a contributor needs that is not obvious from reading the code, and
several things that are actively misleading if you only read the code.

---

## 1. Layout

```
ward_cgm_sim/
  config.py          every tunable and every reward weight, in one place
  core/
    patient.py       PatientState (hidden truth) + PatientKnowledge (what the agent learned)
    glucose.py       latent glucose, the CGM sensing chain, point-of-care testing
    alarms.py        alarm generation, persistence, false-alarm classification
    eligibility.py   inclusion/exclusion criteria and de-enrolment
    staff.py         background staff availability and escalation
    bedflow.py       beds, admissions queue, the discharge pipeline
    ward_map.py      tile grid, bed positions, BFS pathfinding
    actions.py       the 24 discrete actions
    observations.py  the observable slice - the POMDP boundary lives here
    rewards.py       named reward components
    engine.py        one 5-minute tick; orchestrates all of the above
  env.py             Gymnasium wrapper. NATIVE ONLY - see §3
  render/
    assets/          the pixel-art sheets + assets-index.json (the contract)
    sprites.py       loads the sheets; procedural rectangles as fallback
    pygame_renderer.py  top-down map + HUD
  agents/            random and rule-based baselines
scripts/             play, run_baseline, evaluate, train_ppo, build_web
web/main.py          browser entrypoint (pygbag)
```

**Source of truth:** `ward_cgm_sim/` is the only place to edit. `web/ward_cgm_sim/`
is a **generated copy** written by `scripts/build_web.py` on every build and is
gitignored - hand-editing it will be silently overwritten and the change lost.
Same for `web/build/`.

**One tick, in order:** resolve the agent's action → advance physiology →
advance the sensor chain and alarms → advance staff and bed flow → score →
check termination. Actions resolve *before* physiology so that treating a
hypoglycaemic patient this step gets credit before the next glucose sample.

---

## 2. The POMDP boundary

The whole point is that clinical facts cost time to learn.

- `PatientState` is ground truth and is **never** exposed directly.
- `PatientKnowledge` records what the agent has learned and when, so information
  can go stale.
- `observations.py` may read `PatientKnowledge`, never a hidden field. If you
  find yourself reaching for `patient.true_glucose` in that file, stop.
- `-1` is a dedicated **unknown** sentinel, deliberately distinguishable from a
  real value. "I have not looked" must not be confusable with "I looked and the
  answer was no".

**Glucose is a snapshot, not a feed.** `CHECK_DASHBOARD` costs a step and
captures `engine.dashboard_snapshot`; the observation serves that snapshot, aged
by time since the read. An agent that never checks has no glucose information at
all. Snapshots are stamped with `patient_id` because beds get reused.

The same applies to the rule-based baseline in `agents/rule_based.py`. It is a
scientific comparator, not a bot that needs to win: it reads only its own
knowledge record and dashboard snapshot. `tests/test_observability.py` enforces
this both statically and behaviourally.

---

## 3. The web import boundary (breaks the browser build)

The browser bundle ships `core/`, `render/`, `agents/` and `config.py`, and
those must import **standard library plus pygame-ce only**.

- `ward_cgm_sim/env.py` subclasses `gymnasium.Env` and pulls numpy. It is
  **never** imported from `__init__.py` and **never** vendored into `web/`.
- `render/pygame_renderer.py` imports numpy lazily *inside* `to_rgb_array()`
  only, which never runs in the browser.
- `tests/test_web_bundle.py` enforces this in a subprocess with numpy,
  gymnasium and stable-baselines3 blocked. `scripts/build_web.py` runs it and
  fails the build on violation.

---

## 4. Counterfactual integrity - the invariant most easily broken

The telemetry and routine-monitoring arms are only comparable because they
simulate the *same ward*. That relies on common random numbers, partitioned by
scope:

| Stream | Drives |
|---|---|
| `engine.rng` | ward level: patient sampling, arrivals, staff. **Must consume identically in both arms.** |
| `patient.rng` | that patient's physiology, transfers, discharge progression |
| `patient.rng_sensor` | that patient's CGM chain |
| `patient.rng_care` | routine-monitoring checks on that patient |
| `patient.rng_action` | consequences of the agent acting on that patient |

**The invariant:** an intervention on one patient may change how many draws
*that* patient consumes, but must not change any other patient's outcomes.

**How it gets broken.** Every one of these has actually happened here:

- A draw gated behind `if patient.is_enrolled` - only true in one arm, so the
  stream desynchronises.
- A draw taken *inside* a conditional branch, so the number of draws depends on
  state the agent influences. Draw unconditionally, apply conditionally.
- Seeding streams by XOR or multiplication of ids - it **collides**. One
  patient's physiology stream became byte-identical to another's sensor stream.
  Use the domain-separated string seeds in `sample_patient`.
- Putting an action-triggered draw on `rng_care`, letting an agent action shift
  the exogenous comparator.

What is *not* a bug: beds and staff are genuinely shared, so treating one
patient can still delay another through contention, and freeing a bed sooner
really does admit the next patient sooner. Only *spurious* coupling through
randomness is eliminated.

`tests/test_counterfactual_rng.py` guards all of this and is mutation-verified.
Re-run it after touching any `random` call.

---

## 5. Analysis traps

- **Detection means somebody knows**, not that a device fired. An alarm on an
  unread board is not detection.
- **Detection delay is conditional on detection and therefore censored.** An
  episode nobody found contributes no delay. Never report it without the
  detection rate beside it.
- **Pool at the event level.** Averaging per-shift ratios while dropping shifts
  with no events compares different subsets of shifts between arms.
- **The effect estimate is the paired contrast**, resampling matched shift
  pairs. Overlap between two marginal confidence intervals is not a test.
- Episode denominators differ slightly between arms by construction, because
  successful treatment prevents episodes. That is real, not an imbalance.

`tests/test_statistics.py` covers the aggregation and both bootstraps against
synthetic data with known answers.

---

## 6. The art, and the one rule that governs it

`render/assets/` holds five indexed PNGs plus `assets-index.json`, which is the
contract: sprite coordinates, the shared 48-colour palette, and the palette
indices used for recolouring. Read the manifest, do not hard-code offsets.

- **Skin tones and blanket colours are palette regions, not separate sprites.**
  Five skins and eight blankets are produced by repainting three and two
  palette entries respectively. `sprites._recolour` repaints the palette entry
  when the surface is still 8-bit and matches on the baked colour otherwise,
  because the browser's SDL_image build does not always hand back an indexed
  surface. Both routes are load-bearing; do not delete one.
- **The fallback is deliberate.** A missing `assets/` directory drops to the
  procedural rectangles so the simulator still runs. A *corrupt* one raises -
  silently degrading to rectangles is how broken art reaches production.
- **Scaling must stay integer and nearest-neighbour.** `transform.scale`, never
  `smoothscale`; 16px art at 2x. Anything else smears the pixel grid.
- **The map is subject to the POMDP boundary too.** The renderer may draw what
  the agent has *learned* (`patient.knowledge.*`), never hidden truth. Drawing
  `discharge_stage` would show a viewer a fact the policy must spend a step
  acquiring. `tests/test_renderer.py` pins this both statically and
  behaviourally.
- **No `hash()` on a string in the render path.** It is salted per process, so
  it makes two runs of the same seed draw different frames.

The renderer resolves walls and the nurse station from their neighbours,
because `ward_map` only stores six tile codes. Add a tile name and you must add
it to the sheet; `test_the_renderer_asks_for_tiles_that_exist` walks the whole
map and will catch a name that is not there.

---

## 7. The browser bridge

`web/main.py` is the browser entrypoint and owns two things the rest of the
repo does not: the keyboard map, and the readout it publishes to whatever page
is hosting it.

**It had no tests at all until recently, and that is how `D` shipped bound to
`CHECK_DASHBOARD` while the on-screen help promised WASD movement** - the nurse
simply could not walk right. Anything that couples the keys, the help text and
the page's control list is worth a test; `tests/test_web_main.py` has them.

- **`#panel=external` in the URL** tells the demo the page is rendering the
  readout, so it drops its in-canvas HUD and gives the whole canvas to the
  ward. A URL fragment rather than frame detection, because it is
  deterministic and fails safe: no fragment, or an unreadable one, means the
  HUD stays. The page and the demo must agree on that string - a test pins it.
- **`Demo.state()` republishes `visible_alarms()`** and derives no visibility
  rule of its own. The engine already owns that rule, and a second copy would
  be free to drift. The same POMDP boundary applies here as in the renderer:
  the page may be told what the agent has learned, never hidden truth.
- **Sent is not delivered.** `postMessage` does not raise when the window it
  reaches is not at the origin it was addressed to, so a send can succeed while
  nothing arrives - which is exactly what happened, leaving the page blank and
  the demo convinced it had done its job. The page acknowledges, and only the
  acknowledgement sets `published_ok`. Until then the canvas draws a compact
  readout and says why. **A broken channel must degrade to a smaller readout,
  never to none.**
- **The origin is pinned** to the hosting page, never `"*"`. The demo lives on
  its own origin precisely so the third-party WASM runtime cannot reach the
  portfolio; this one-way channel is the only thing that crosses, and the page
  treats everything arriving as hostile input.

Two files in two repositories have to agree about all of this. The tests read
the page directly and check the fragment, and check that every id its script
looks up exists in its markup - the readout was silently dropped for a while
because a restructure removed an id the listener guarded on.

---

## 8. Calibration status

**No parameter in this model is derived from primary data.** The population is
deliberately configured to increase eligible-patient and event counts, because
a realistic cohort produces almost no events in a 12-hour shift and any arm
difference is lost in noise.

Absolute rates are therefore **not** incidence estimates. Only the contrast
between arms is interpretable, and only as a statement about the model.
`docs/POMDP.md` §14 lists exactly which inputs must be sourced before this
informs a real study design; `usual_care.routine_detection_prob` is the single
most influential.

---

## 9. Known unfinished work

- **Runtime vendoring** (`scripts/build_web.py --vendor-runtime`) downloads the
  ~25 MB pygbag runtime to serve it same-origin, which would remove the
  third-party code dependency. It **does not boot**, and its URL handling still
  assumes the 0.9.x `/archives/` layout while 0.9.3 emits `/cdn/<version>/`.
  Both must be fixed before enabling it. The shipped mitigation is that the demo
  is served from a separate origin instead.
- **Art** is delivered and wired in. A few tiles in the sheet have no map data
  behind them yet (`bedside_cabinet`, `iv_drip_stand`, `curtain_open/closed`,
  `visitor_chair`, `hand_wash_basin`, `alcohol_gel_dispenser`,
  `clinical_waste_bin`, `store_cupboard`, `sluice_dirty_utility_door`,
  `bay_threshold`, `entrance_mat`, `drug_room_floor`) and two overlays
  (`point_of_care_test`, `treatment_given`) need a transient event feed the
  renderer does not have. Placing furniture would need `ward_map` to carry it,
  otherwise the agent walks through chairs.
- **No trained policy.** `scripts/train_ppo.py` is a worked example, not a
  result. Nothing in this repository has learned anything: the demo is driven
  by the rule-based comparator, and the project page says so plainly. Do not
  let any copy imply otherwise.

---

## 10. Deployment facts

- pygbag must be **≥ 0.9.3**. 0.9.2 fails to boot over HTTP/2 with
  `Cannot read properties of undefined (reading 'statSync')` - it works locally
  over HTTP/1.0 and fails once deployed, so local testing will not catch it.
- The demo **cannot run in an iframe sandboxed without `allow-same-origin`**;
  it loses the storage access BrowserFS needs and hangs. Adding that flag makes
  the sandbox pointless. A separate origin is the answer.
- The demo is deployed to `ward-cgm-demo.web.app`, separate from the portfolio
  site. Deploy commands are in the parent repository's `CLAUDE.md`.
