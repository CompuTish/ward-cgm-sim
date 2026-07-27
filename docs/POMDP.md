# Formal model: inpatient CGM telemetry as a POMDP

> **Academic model.** Every clinical value here is a simplified, configurable,
> guideline-inspired placeholder chosen to make the simulation behave
> plausibly. None of it is prescribing guidance, bedside instruction, or
> clinical decision support. All parameters live in `ward_cgm_sim/config.py`
> so that any of them can be varied for sensitivity analysis.

## 1. The decision problem

The agent is a **ward nurse or shift coordinator** on a 32-bed mixed
medical/surgical ward, managing continuous glucose monitoring (CGM) telemetry
as part of **routine ward care**. There is no separate research team: enrolment
paperwork, alarm response, escalation, sensor troubleshooting and discharge work
all compete for one person's time across a single 12-hour shift. That
competition is the workflow question the simulator exists to probe.

It is a **partially observable** Markov decision process because the agent
cannot see what matters without spending time: a patient's insulin regimen is
in the drug chart, their capacity to consent is established by talking to them,
their true glucose is only knowable by testing, and whether a colleague is free
is only knowable by asking.

## 2. Formal definition

The model is the 7-tuple ⟨S, A, T, R, Ω, O, γ⟩.

| Symbol | Meaning in this model |
|---|---|
| **S** | Full ward state: every patient's latent clinical state, sensor state and knowledge record; staff availability; the admissions queue; the shift clock; the agent's position. |
| **A** | `Discrete(24)` — 4 movement + 20 interaction actions (§4). |
| **T** | Stochastic transition over one 5-minute step (§5). |
| **R** | Weighted sum of named components, safety-dominant (§6). |
| **Ω** | The observation set: a 296-dimensional real vector (§3). |
| **O** | The observation function: what is visible from where the agent stands, plus what it has previously learned and recorded. |
| **γ** | 1.0 — finite horizon of 144 steps, so no discounting is needed. |

**Episode** = one 12-hour shift = 144 steps of 5 minutes.

**Termination**: shift end (truncation), unsafe overcrowding, or a serious
adverse event. The headline metric is the proportion of **shifts completed
without a serious incident**.

## 3. Observation space

`Box(low=-1.0, high=1.5, shape=(8 + 9 × 32,)) = (296,)`

`-1` is a dedicated **unknown** sentinel. It is deliberately distinguishable
from any real value, so the agent can tell "I have not looked" apart from "I
looked and the answer was no". That distinction is what makes this a POMDP
rather than an MDP with noisy inputs.

**Ward-level (8 values)**

| # | Feature |
|---|---|
| 0 | Fraction of the shift elapsed |
| 1 | Bed occupancy fraction |
| 2 | Free beds fraction |
| 3 | ED/admissions queue length, scaled |
| 4 | Count of visible alarms, scaled |
| 5 | Coarse staff availability (0 skeleton / 0.5 stretched / 1 comfortable) |
| 6–7 | Agent x, y position |

**Per bed (9 values × 32 beds)**

| # | Feature | Visible when |
|---|---|---|
| 0 | Occupied | always |
| 1 | Patient visibly at the bed | always (0 if off ward or walking) |
| 2 | Enrolment status | always |
| 3 | Known eligibility | only after `REVIEW_NOTES` |
| 4 | Known consent status | only after `ASK_CONSENT` |
| 5 | CGM value | only if enrolled, telemetry on, and signal present |
| 6 | Steps since a valid CGM reading | only if enrolled |
| 7 | Alarm severity for this bed | only if the dashboard has been read |
| 8 | Known discharge readiness | only after review |

**Hidden throughout** (never in the observation): true eligibility, insulin
injections per day, capacity, consent disposition, expected length of stay,
true glucose, true discharge readiness, hypoglycaemia and hyperglycaemia risk,
sensor accuracy and bias, whether the insulin regimen has changed, whether the
patient has become ineligible, and whether a given staff role is actually free.

The **telemetry dashboard is positional**: alarms are visible while standing at
the nurse station, or from anywhere for alarms raised before the last
`CHECK_DASHBOARD`. Walking away and never looking back means never learning
that somebody is alarming.

## 4. Action space

`Discrete(24)`.

**Movement (4)** — `MOVE_UP/DOWN/LEFT/RIGHT`, one tile per step.

**Interactions (20)** — applied to the bed the agent is standing next to;
dashboard actions require the nurse station.

| Action | Effect |
|---|---|
| `CHECK_DASHBOARD` | Refresh the alarm picture |
| `CHECK_PATIENT` | Bedside look; reveals obvious discharge readiness |
| `REVIEW_NOTES` | Reveal the eligibility criteria from the notes/drug chart |
| `ASK_CONSENT` | Seek verbal consent (may be declined; invalid without capacity) |
| `ENROL` | Fit telemetry |
| `REVIEW_ELIGIBILITY` | Re-check an enrolled patient against the criteria |
| `DEENROL` | Remove telemetry |
| `RESPOND_ALARM` | Acknowledge an alarm |
| `POC_GLUCOSE_TEST` | Confirmatory capillary blood glucose |
| `TREAT_HYPO` / `TREAT_HYPER` | Simplified placeholder pathways |
| `ESCALATE` | Refer to the medical/diabetes pathway |
| `ASK_HELP_HCA/NURSE/DOCTOR/SURGEON` | Request help from a specific role |
| `TROUBLESHOOT_SENSOR` | Attempt to restore a failed sensor |
| `SUPPORT_DISCHARGE` | Advance the discharge pipeline |
| `PRIORITISE_BEDFLOW` | Chase the bed-flow backlog |
| `WAIT` | Do nothing |

Two design notes. **Avoiding an inappropriate enrolment** is not an action: it
is expressed by *not* choosing `ENROL`, which is penalised when wrong — so
avoidance is learned behaviour rather than a button. **Asking for help is split
by role** so that choosing the right colleague is itself a decision; asking the
wrong role wastes the step.

## 5. Transition dynamics

**Glucose.** A mean-reverting process toward each patient's usual level, plus
meal and insulin events that ramp over 30 and 60 minutes respectively (never as
instantaneous jumps — an insulin dose does not move glucose 2 mmol/L in five
minutes, and modelling it that way makes the trajectory untrackable by any
sensor). Deterioration episodes drift a patient toward hypo- or hyperglycaemia
with probability scaled by their individual risk.

**CGM sensing.** Three layers, deliberately distinct:

1. `true_glucose` — latent, never observed by anyone.
2. **CGM** — the lagged (~10 min), biased, noisy value on the dashboard.
   Each sensor carries a fixed calibration bias drawn at insertion; noise
   triples when a sensor degrades; transient artefacts produce false alarms.
3. **Point-of-care capillary** — small error, and **trusted over CGM** wherever
   the two conflict.

**Sensor failure is silent.** Signal loss produces *no alarm* — the data simply
stops arriving, and the only cue is a growing "steps since last reading" gap.

**Staff.** Each role is a hidden two-state Markov chain. The agent sees only a
coarse ward-level summary and must ask to discover whether a specific role is
free.

**Bed flow.** Arrivals follow a time-varying Poisson process into the ED queue.
Discharge readiness is a function of progress through the patient's *expected*
length of stay, which keeps the model self-consistent: expected stay is an
enrolment criterion, so a patient documented for 48 hours cannot evaporate an
hour later. The pipeline runs `NOT_READY → READY → REVIEWED → SUPPORTED →
DISCHARGED`; background staff advance it slowly on their own, and the agent's
involvement accelerates it.

### Common random numbers

The telemetry-on and telemetry-off arms share a seed across **partitioned**
streams, so both arms simulate the *same ward*:

- `engine.rng` — ward level (patient sampling, arrivals, staff). Consumes
  identically in both arms.
- `PatientState.rng` / `.rng_sensor` / `.rng_care` / `.rng_action` — that
  patient's physiology and flow, their CGM chain, routine checks on them, and
  consequences of the agent acting on them.

**Invariant:** an intervention on one patient may change how many draws *that*
patient consumes, but must not change any other patient's random outcomes. This
is enforced by `tests/test_counterfactual_rng.py`. Shared *resources* (beds,
staff) still couple patients — that is a real ward effect and is meant to
remain; only spurious coupling through randomness is eliminated.

## 6. Reward function

Weights are in `config.RewardConfig`. Safety dominates by an order of
magnitude.

**Large positive**

| Component | Weight |
|---|---|
| Hypoglycaemia treated promptly | +10 (×2 if severe) |
| Correct escalation of severe/recurrent events | +8 |
| Hypoglycaemia caught pre-emptively | +6 |
| Shift completed with no serious adverse event | +15 |
| Enrolled patient safe, per patient per step | +0.02 |
| Safe occupancy, per step | +0.05 |

**Moderate positive**

| Component | Weight |
|---|---|
| Correct enrolment | +5 |
| Correct de-enrolment | +4 |
| Discharge supported | +3 |
| Alarm-fatigue avoidance (end of shift, scaled) | up to +3 |
| Fast alarm response | +2 |
| Correctly identifying an ineligible patient | +2 |
| Queue reduced, per patient | +0.5 |

**Negative**

| Component | Weight |
|---|---|
| Serious adverse event | −50 (terminal) |
| Unsafe overcrowding | −30 (terminal) |
| Missed severe hypoglycaemia | −20 |
| Enrolling an ineligible patient | −6 |
| Wrong-patient treatment | −4 |
| Unnecessary de-enrolment | −4 |
| Missed eligible patient | −3 |
| Unnecessary treatment when point-of-care is normal | −3 |
| Treating without point-of-care confirmation | −2 |
| Unsafe prioritisation during an urgent alarm, per step | −1 |
| Time below range, per patient per step | −0.5 |
| Ignored signal loss, per patient per step | −0.3 |
| Wrong-role help request | −0.3 |
| Delayed alarm response, per step | −0.2 |
| Failure to de-enrol, per patient per step | −0.2 |
| Excessive staff workload | −0.2 |
| Discharge delay, per patient per step | −0.1 |
| Queue pressure, per patient per step | −0.05 |
| Bed overcrowding, per step | −1 |
| Invalid action | −0.05 |

## 7. Patient state model

Hidden: diabetes type (type 1 / 2 / 3c / other / none), insulin injections per
day, capacity, consent disposition, expected length of stay, specialty
(medical/surgical), pregnancy or breastfeeding, end-of-life status,
hypoglycaemia and hyperglycaemia risk, usual glucose level, true glucose,
individualised alarm threshold, true discharge readiness.

Visible: bed, location (in bed / walking / off ward), enrolment status, and any
fact the agent has *learned*, recorded in `PatientKnowledge` with the step it
was learned — so information can go stale.

Patients move visibly on the map for admissions, transfers and discharges.

## 8. Alarm generation

| Alarm | Condition |
|---|---|
| Hypoglycaemia | CGM < **3.9 mmol/L** |
| Severe hypoglycaemia | CGM < **3.0 mmol/L** (stronger penalty) |
| Hyperglycaemia | CGM > **14.0 mmol/L**, or a per-patient individualised threshold (default 18.0) |
| Rapid fall | smoothed 15-minute trend ≤ −1.8 mmol/L |
| Rapid rise | smoothed 15-minute trend ≥ +2.6 mmol/L |

- **Persistence**: an out-of-range value must repeat before alarming, which
  suppresses single-sample artefacts. This is the main lever for trading alarm
  burden against detection latency.
- **Individualised hyperglycaemia thresholds** exist for patients with chronic
  uncontrolled hyperglycaemia — the alarm-fatigue lever. Without them, those
  patients alarm constantly and the board becomes noise.
- **Trend alarms are computed on a smoothed signal**; differencing raw samples
  doubles the noise and makes trend alarms fire on nothing.
- **A false alarm is only counted when the latent glucose is clearly on the
  other side of the threshold** (0.8 mmol/L margin), so a genuinely borderline
  patient is not scored as nuisance.
- **Signal loss raises nothing.**

## 9. Eligibility and de-enrolment

**Inclusion — all must hold:** diabetes of any type; ≥2 insulin injections per
day; expected ward stay ≥48 hours; capacity to give verbal informed consent;
consent given.

**Exclusion — any excludes:** <2 injections per day; expected stay <48 hours;
lacks capacity; declines; pregnancy or breastfeeding; end-of-life care.

**Later ineligibility:** insulin reduced to once daily, transition to
end-of-life care, a revised discharge plan taking the stay under 48 hours, or
withdrawal of consent. Any of these requires de-enrolment; failing to
de-enrol accrues a per-step penalty, and de-enrolling somebody who still
qualifies is penalised too.

Consent is deliberately **not** re-tested for already-enrolled patients — they
have consented, and re-testing would flag the whole cohort.

## 10. Staff and escalation

Roles: HCA, nurse, doctor, surgeon, diabetes team. Each has a competency set;
asking the wrong role for a task yields nothing and costs the step. `ESCALATE`
routes severe or recurrent glycaemic events to the diabetes team, falling back
to the on-call doctor. Surgical patients route to surgeons for discharge
decisions. Requests accumulate a workload measure; exceeding a threshold is
penalised.

## 11. Bed flow and discharge

32 beds; an ED/admissions queue; time-varying arrivals; transfers off ward
(imaging, theatre) that make a patient temporarily unavailable; the four-stage
discharge pipeline; overcrowding penalties above a queue threshold; and
termination on unsafe overcrowding.

## 12. Usual care — the comparator

Critically, the counterfactual is **telemetry versus routine monitoring**, not
telemetry versus nothing. Every patient — including those on CGM — receives
routine capillary rounds and symptom recognition, modelled in
`config.UsualCareConfig` at roughly 4–6 hourly detection probability, higher
when a patient is symptomatically severe. CGM is *additive* to standard care.

Without this, "no telemetry" would mean "no monitoring at all" and the model
would flatter CGM enormously. The question the simulator actually asks is
whether the alarm gets there **first, and by how much**.

## 13. Testing whether CGM telemetry improves ward workflow

`scripts/evaluate.py` runs matched-seed batches with `telemetry_enabled`
True and False and compares:

**Primary** — mean hypoglycaemia **detection delay** (steps below 3.9 mmol/L
before anybody knows) and detection rate, both restricted to the *monitored
cohort* so the arms are like-for-like. Ward-wide figures are diluted by the
majority of patients who are never eligible.

**Secondary** — time below range, severe hypoglycaemia events, serious adverse
events, incident-free shift rate, alarm burden and false-alarm rate, enrolment
precision and recall, discharge delay, queue length, overcrowding.

A hypoglycaemic **event** follows the consensus definition: at least 15 minutes
(3 steps) below threshold. Without that, every transient dip inflates the
denominator and the detection rate becomes meaningless.

**What this can and cannot show.** It is a simulation study of a workflow
model. It cannot demonstrate clinical benefit. It can show whether a modelled
mechanism plausibly produces one, where the effect is sensitive to assumptions,
and which parameters matter most — which is exactly what is useful before
designing a real study.

## 14. Calibration and limitations

**The population is deliberately signal-enriched and is not an epidemiological
model of a real ward.** This is the most important caveat in the document.

| Parameter | Model | A real acute ward | Why |
|---|---|---|---|
| Diabetes prevalence | 45% | ~20% | Enriched for signal density |
| On ≥2 insulin injections/day | 72% of those | ~30% of those | Enriched |
| Eligible patients per 32 beds | ~4–5 | ~1–2 | Consequence of the above |
| Length of stay | Conditional on insulin status | Correlated in reality too | Resolves a real tension (below) |
| Severe hypoglycaemia | ~2 events/shift | Far rarer | Follows from the enriched cohort |

With a realistic population, one or two eligible patients per ward produce
almost no events in a 12-hour shift, and any arm difference would be swamped by
noise at any tractable number of episodes. The cohort is therefore enriched so
the *mechanism* is observable.

**Consequence for interpretation: absolute rates from this model are not
estimates of real-ward incidence.** Only the *contrast* between the telemetry
and routine-monitoring arms is meaningful, and even that is a statement about
the model, not about patients.

**Length of stay is drawn conditional on insulin status.** Insulin-treated
inpatients genuinely have longer admissions, and modelling that resolves a
tension that would otherwise be unresolvable: a uniformly long-stay ward gives
a workable telemetry cohort but almost no discharges, while a uniformly
short-stay ward turns over briskly and leaves nobody eligible for enrolment.

**Other parameters that carry real uncertainty and should be varied in any
sensitivity analysis:**

- `usual_care.routine_detection_prob` (0.02/step) — the single most influential
  parameter in the whole comparison. It sets how good the comparator is, and
  therefore the size of any telemetry effect. It is a smooth per-step hazard
  standing in for what is really a *scheduled* round; modelling actual 4–6
  hourly checks would change the shape of the detection-delay distribution.
- `alarms.persistence_readings` and `false_alarm_margin` — trade alarm burden
  against detection latency directly.
- `glucose.cgm_noise_sd`, `cgm_bias_sd`, `cgm_lag_steps` — sensor performance,
  which should be set from the accuracy data of whichever device is being
  modelled rather than left at these placeholder values.
- `patients.initial_enrolled_fraction` — how much of the cohort is inherited at
  handover versus recruited during the shift.

**Known simplifications.** Glucose is a single scalar with no insulin
pharmacokinetics; treatment effects are fixed ramps rather than dose-dependent;
staff availability is a two-state chain with no handover, breaks or skill mix;
patients do not deteriorate for non-glycaemic reasons; and the ward has no
night/day staffing difference within the shift.
