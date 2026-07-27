"""The counterfactual-integrity invariant.

The telemetry and routine-monitoring arms are only comparable if they simulate
the *same ward*. That requires common random numbers partitioned by scope, so
that an intervention on one patient cannot shift any other patient's
trajectory. These tests exist because that property is easy to break by
accident - a single draw placed behind ``if patient.is_enrolled`` is enough,
and nothing else in the suite would notice.
"""

import random

import pytest

from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.actions import Action
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.core.patient import DischargeStage

SEEDS = [0, 1, 7, 42]


def _ward_stream_state(engine: WardEngine):
    return engine.rng.getstate()


def _snapshot(patient):
    return (
        round(patient.true_glucose, 9),
        patient.discharge_stage,
        patient.location,
        patient.insulin_injections_per_day,
    )


def _patient_trajectories(engine: WardEngine):
    """Every patient's clinical course, keyed by patient id.

    Deliberately the *trajectory* rather than the RNG state. Comparing stream
    states alone is not a real check: a draw wrongly taken from the shared ward
    stream leaves every per-patient stream untouched, so a stream-state
    assertion sails straight past the bug it is supposed to catch.

    Discharged patients are included. Restricting the comparison to patients
    still on the ward would let the worst case - a spurious change that
    discharged somebody early - vanish from the set being compared.
    """
    states = {p.patient_id: _snapshot(p) for p in engine.flow.patients()}
    for p in engine.flow.discharged:
        states[p.patient_id] = _snapshot(p)
    return states


@pytest.mark.parametrize("seed", SEEDS)
def test_ward_stream_identical_across_arms(seed):
    """The ward-level stream must consume identically with telemetry on/off.

    Positive control: assert the ward is actually populated, so that an engine
    which silently produced no patients cannot pass by vacuous agreement.
    """
    engines = {}
    for telemetry in (True, False):
        cfg = SimConfig()
        cfg.telemetry_enabled = telemetry
        engines[telemetry] = WardEngine(cfg, seed=seed)

    assert engines[True].flow.occupied_beds > 20, "ward should be populated"

    for step in range(SimConfig().steps_per_episode):
        done = False
        for telemetry in (True, False):
            engine = engines[telemetry]
            # A fixed, non-trivial action sequence that touches bed flow.
            action = Action.PRIORITISE_BEDFLOW if step % 7 == 0 else Action.WAIT
            _obs, _r, terminated, truncated, _info = engine.step(action)
            done = done or terminated or truncated
        assert _ward_stream_state(engines[True]) == _ward_stream_state(engines[False]), (
            f"ward-level RNG diverged between arms at step {step}"
        )
        if done:
            break


@pytest.mark.parametrize("seed", SEEDS)
def test_intervening_on_one_patient_does_not_perturb_others(seed):
    """Supporting a discharge must not alter any other patient's stream.

    This is the exact failure mode that a single shared stream produced: the
    intervention changed which branch consumed a draw, shifting every
    subsequent patient's trajectory while the output still looked plausible.
    """
    cfg_a, cfg_b = SimConfig(), SimConfig()
    control = WardEngine(cfg_a, seed=seed)
    treated = WardEngine(cfg_b, seed=seed)

    # Force a deterministic, always-available target rather than waiting for one
    # to arise: a runtime skip would let this test silently stop running.
    target_bed = min(p.bed for p in treated.flow.patients())
    for engine in (control, treated):
        patient = engine.flow.patient_at_bed(target_bed)
        patient.discharge_stage = DischargeStage.REVIEWED
        patient.knowledge.known_discharge_ready = True

    # Capture the target's identity BEFORE intervening; bed turnover during the
    # comparison window could otherwise put a different patient in that bed.
    target_patient = treated.flow.patient_at_bed(target_bed).patient_id

    # The cohort to compare is fixed at intervention time. Patients admitted
    # *afterwards* are excluded on purpose: discharging somebody frees a bed
    # sooner, so the next admission legitimately happens sooner. That is a real
    # ward-capacity effect and is meant to be modelled. What this test targets
    # is *spurious* coupling - a patient's random outcomes changing for no
    # reason but another patient's treatment.
    cohort = {p.patient_id for p in treated.flow.patients()}

    # Positive control: both arms agree before the intervention.
    assert _patient_trajectories(control) == _patient_trajectories(treated)

    approach = treated.ward_map.approach_tile(target_bed)
    treated.agent_x, treated.agent_y = approach
    treated.step(Action.SUPPORT_DISCHARGE)
    control.step(Action.WAIT)

    # Compare at EVERY step, not just a final snapshot: a divergence that
    # appears and then washes out would be invisible in an end-state check.
    compared_steps = 0
    for _ in range(30):
        if control.terminated or control.truncated or treated.terminated or treated.truncated:
            break
        control.step(Action.WAIT)
        treated.step(Action.WAIT)
        compared_steps += 1

        control_states = _patient_trajectories(control)
        treated_states = _patient_trajectories(treated)
        shared = (set(control_states) & set(treated_states) & cohort) - {target_patient}
        assert len(shared) > 10, "positive control: substantial cohort to compare"

        perturbed = [
            pid for pid in shared if control_states[pid] != treated_states[pid]
        ]
        assert not perturbed, (
            f"intervening on bed {target_bed} altered the clinical course of "
            f"{len(perturbed)} other patient(s) {sorted(perturbed)} after "
            f"{compared_steps} steps. A draw is shared across patients."
        )

    assert compared_steps >= 10, "positive control: enough steps actually compared"


def _immutable_profile(patient):
    """Characteristics fixed at sampling time and never modified afterwards.

    Deliberately excludes anything a shift can change (insulin frequency,
    end-of-life status, glucose), so this compares *who the patient is* rather
    than what has happened to them.
    """
    return (
        patient.diabetes_type,
        patient.specialty,
        round(patient.expected_los_hours, 9),
        patient.has_capacity,
        patient.will_consent,
        patient.pregnant_or_breastfeeding,
        round(patient.hypo_risk, 9),
        round(patient.hyper_risk, 9),
        round(patient.target_glucose, 9),
    )


def _all_people(engine: WardEngine):
    """Immutable profiles for everyone the episode has generated so far.

    Spans occupied beds, the admissions queue and discharged patients. A map
    built from beds alone silently omits a patient who has been admitted in one
    arm but is still waiting in the other - which is exactly the population the
    capacity check needs to compare.
    """
    people = {p.patient_id: _immutable_profile(p) for p in engine.flow.patients()}
    for p in engine.flow.queue:
        people[p.patient_id] = _immutable_profile(p)
    for p in engine.flow.discharged:
        people[p.patient_id] = _immutable_profile(p)
    return people


def test_capacity_pathway_does_not_disturb_the_ward_stream():
    """Freeing a bed earlier must change only *timing*, not the ward draw stream.

    The fixed-cohort comparison in the test above deliberately excludes
    patients admitted after the intervention, because their clinical course
    legitimately differs once a bed frees sooner. That exclusion would hide a
    regression which consumed `engine.rng` only when an earlier admission
    happens - so the capacity pathway gets its own check here.

    What must hold: the ward-level stream consumes identically in both arms,
    and any given patient id is the *same person* in both arms. What is
    allowed to differ: when that person is admitted.
    """
    def build():
        cfg = SimConfig()
        cfg.ward.initial_occupancy = 1.0  # full ward, so the queue cannot drain
        return WardEngine(cfg, seed=4)

    control, treated = build(), build()

    assert control.flow.free_beds == 0, "positive control: ward must start full"
    assert control.flow.queue_length > 0, "positive control: patients must be waiting"

    target_bed = min(p.bed for p in treated.flow.patients())
    for engine in (control, treated):
        patient = engine.flow.patient_at_bed(target_bed)
        patient.discharge_stage = DischargeStage.REVIEWED

    treated.agent_x, treated.agent_y = treated.ward_map.approach_tile(target_bed)
    treated.step(Action.SUPPORT_DISCHARGE)
    control.step(Action.WAIT)

    # Check immediately, before advancing any further: a divergence that appeared
    # here and reconverged by the next step would otherwise slip through.
    assert control.rng.getstate() == treated.rng.getstate(), (
        "ward-level RNG diverged on the intervention step itself"
    )

    timing_differed = False
    for step in range(40):
        control.step(Action.WAIT)
        treated.step(Action.WAIT)

        # Admission *timing* is what the intervention changes. Totals converge
        # again once background discharges catch up, so comparing only the
        # final counts would let this positive control pass vacuously.
        if treated.flow.total_admissions != control.flow.total_admissions:
            timing_differed = True

        assert control.rng.getstate() == treated.rng.getstate(), (
            f"ward-level RNG diverged at step {step}: an earlier admission is "
            f"consuming draws from the shared ward stream"
        )

        # Populations must span beds, queue AND discharged patients. Comparing
        # occupied beds alone would skip precisely the patient this test exists
        # for: the one already admitted in the treated arm but still queued in
        # the control arm.
        control_people = _all_people(control)
        treated_people = _all_people(treated)
        shared = set(control_people) & set(treated_people)
        assert len(shared) > 20, "positive control: substantial population compared"
        for pid in shared:
            assert control_people[pid] == treated_people[pid], (
                f"patient {pid} is a different person in the two arms; the ward "
                f"stream is producing different patients"
            )

    # Positive control: the intervention must actually have changed admission
    # timing at some point, or this test proves nothing at all.
    assert timing_differed, (
        "intervention never changed admission timing, so the capacity pathway "
        "was never exercised and this test proves nothing"
    )


STREAM_NAMES = ("rng", "rng_sensor", "rng_care", "rng_action")


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_no_stream_collisions_across_patients_or_domains(seed):
    """Every (patient, stream) pair must be an independent sequence.

    Arithmetic seeding (XOR/multiply of ids) silently collides - it gave one
    patient's physiology stream the identical state to another's sensor
    stream, correlating things that must be independent. Compares across BOTH
    axes, patients and domains, not just one.
    """
    engine = WardEngine(SimConfig(), seed=seed)
    patients = list(engine.flow.patients()) + list(engine.flow.queue)
    assert len(patients) > 20, "positive control: ward should be populated"

    # Distinct objects
    objects = {id(getattr(p, name)) for p in patients for name in STREAM_NAMES}
    assert len(objects) == len(patients) * len(STREAM_NAMES), "streams are shared objects"

    # Distinct sequences: take several draws so a coincidental first-value
    # match cannot mask a genuine collision.
    signatures = {}
    for p in patients:
        for name in STREAM_NAMES:
            stream = getattr(p, name)
            signature = tuple(stream.random() for _ in range(5))
            key = (p.patient_id, name)
            assert signature not in signatures, (
                f"stream collision: {key} produces the same sequence as "
                f"{signatures[signature]}"
            )
            signatures[signature] = key


def test_treating_one_patient_does_not_change_another_patients_poc():
    """The action-consequence stream must be per patient.

    A shared action stream means giving patient A a point-of-care test shifts
    the measurement error patient B gets on theirs - a spurious coupling that
    would quietly bias any comparison involving differing action counts.
    """
    def poc_for_b(also_test_a: bool) -> float:
        engine = WardEngine(SimConfig(), seed=5)
        beds = sorted(p.bed for p in engine.flow.patients())
        bed_a, bed_b = beds[0], beds[1]

        # Step count is held constant across both runs, so patient B's
        # physiology has advanced identically. The ONLY difference is whether
        # patient A consumed an action-consequence draw first.
        if also_test_a:
            engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(bed_a)
            engine.step(Action.POC_GLUCOSE_TEST)
        else:
            engine.step(Action.WAIT)

        engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(bed_b)
        engine.step(Action.POC_GLUCOSE_TEST)

        patient_b = engine.flow.patient_at_bed(bed_b)
        assert patient_b.knowledge.last_poc_value is not None, (
            "positive control: patient B must actually have been tested"
        )
        return patient_b.knowledge.last_poc_value

    assert poc_for_b(False) == pytest.approx(poc_for_b(True))


def test_seeding_is_reproducible():
    """The same seed must reproduce the same episode exactly."""
    def run(seed):
        engine = WardEngine(SimConfig(), seed=seed)
        rewards = []
        rng = random.Random(99)
        while True:
            _o, r, t, tr, _i = engine.step(rng.randrange(len(Action)))
            rewards.append(r)
            if t or tr:
                break
        return rewards

    first = run(11)
    assert len(first) > 10, "positive control: episode should run for many steps"
    assert first == run(11)
