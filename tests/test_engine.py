"""Engine invariants: episode structure, termination, and reward wiring.

These are the properties that must hold for the simulator to be a coherent
POMDP at all, independent of whether its clinical numbers are well calibrated.
"""

import pytest

from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.actions import ADMIN_ACTIONS, Action
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.core.observations import UNKNOWN, observation_size
from ward_cgm_sim.core.patient import DischargeStage, EnrolmentStatus


def run_to_end(engine, action=Action.WAIT, limit=200):
    steps = 0
    while steps < limit:
        _o, _r, terminated, truncated, info = engine.step(int(action))
        steps += 1
        if terminated or truncated:
            return steps, info
    raise AssertionError("episode failed to terminate within the step limit")


# ---------------------------------------------------------------------------
# Episode structure
# ---------------------------------------------------------------------------

def test_shift_is_144_five_minute_steps():
    cfg = SimConfig()
    assert cfg.steps_per_episode == 144
    assert cfg.minutes_per_step == 5
    assert cfg.steps_per_episode * cfg.minutes_per_step == 12 * 60


def test_ward_starts_with_32_beds():
    engine = WardEngine(SimConfig(), seed=0)
    assert engine.flow.n_beds == 32
    assert engine.flow.occupied_beds > 0, "positive control: ward should be populated"
    assert engine.flow.occupied_beds <= 32


def test_episode_terminates_and_reports_a_reason():
    engine = WardEngine(SimConfig(), seed=1)
    steps, info = run_to_end(engine)
    assert 0 < steps <= 144
    assert info["termination_reason"] in {
        "shift_end",
        "unsafe_overcrowding",
        "serious_adverse_event",
    }


def test_unsafe_overcrowding_terminates_the_shift():
    cfg = SimConfig()
    engine = WardEngine(cfg, seed=2)
    # Force the queue past the unsafe threshold.
    waiting = engine.flow.queue[0] if engine.flow.queue else None
    if waiting is None:
        pytest.skip("seed produced no queued patient to clone")
    engine.flow.queue.extend([waiting] * cfg.ward.unsafe_queue_length)
    _o, _r, terminated, _tr, info = engine.step(int(Action.WAIT))
    assert terminated
    assert info["termination_reason"] == "unsafe_overcrowding"


def test_serious_adverse_event_terminates_the_shift():
    cfg = SimConfig()
    engine = WardEngine(cfg, seed=3)
    patient = next(engine.flow.patients())
    patient.true_glucose = 2.0
    patient.severe_hypo_untreated_steps = cfg.usual_care.sae_untreated_steps - 1
    patient.hypo_risk = 1.0
    # Suppress the routine-monitoring rescue so the event is allowed to mature.
    cfg.usual_care.routine_detection_prob = 0.0
    cfg.glucose.reversion_rate = 0.0
    cfg.glucose.process_noise = 0.0

    terminated = False
    for _ in range(4):
        _o, _r, terminated, _tr, info = engine.step(int(Action.WAIT))
        if terminated:
            break
    assert terminated, "untreated severe hypoglycaemia should end the shift"
    assert info["termination_reason"] == "serious_adverse_event"
    assert info["kpi"]["serious_adverse_events"] >= 1


# ---------------------------------------------------------------------------
# Observations - the partially observable part
# ---------------------------------------------------------------------------

def test_observation_has_the_declared_length():
    engine = WardEngine(SimConfig(), seed=0)
    assert len(engine.observation()) == observation_size(32)


def test_hidden_eligibility_is_unknown_until_the_notes_are_read():
    """The core POMDP property: clinical facts are not free."""
    engine = WardEngine(SimConfig(), seed=4)
    bed = min(p.bed for p in engine.flow.patients() if not p.is_enrolled)

    from ward_cgm_sim.core.observations import BED_FEATURES, WARD_FEATURES

    def known_eligibility():
        obs = engine.observation()
        return obs[WARD_FEATURES + bed * BED_FEATURES + 3]

    assert known_eligibility() == UNKNOWN, "eligibility should start unknown"

    engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(bed)
    engine.step(int(Action.REVIEW_NOTES))
    assert known_eligibility() != UNKNOWN, "reviewing notes should reveal eligibility"


def test_dashboard_is_not_visible_from_across_the_ward():
    """Alarms are only on the board; the agent must look."""
    engine = WardEngine(SimConfig(), seed=5)
    # Stand somewhere away from the nurse station and never check the board.
    engine.agent_x, engine.agent_y = 1, 1
    engine.dashboard_seen_step = None
    assert engine.visible_alarms() == []


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def test_reward_components_are_named_and_populated():
    engine = WardEngine(SimConfig(), seed=6)
    _steps, info = run_to_end(engine)
    components = info["reward_components"]
    assert components, "positive control: some reward must have been accrued"
    assert all(isinstance(v, float) for v in components.values())


def test_enrolling_an_ineligible_patient_is_penalised():
    engine = WardEngine(SimConfig(), seed=7)
    patient = next(
        p for p in engine.flow.patients() if not p.is_enrolled and not p.has_diabetes
    )
    patient.consent_asked = True
    patient.will_consent = True
    patient.consent_declined = False

    engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(patient.bed)
    _o, reward, _t, _tr, info = engine.step(int(Action.ENROL))

    assert info["kpi"]["incorrect_enrolments"] == 1
    assert info["reward_components"]["enrolled_ineligible"] < 0
    assert reward < 0


def test_correctly_enrolling_an_eligible_patient_is_rewarded():
    """Positive counterpart to the penalty above."""
    from ward_cgm_sim.core.patient import DiabetesType

    engine = WardEngine(SimConfig(), seed=8)
    patient = next(p for p in engine.flow.patients() if not p.is_enrolled)
    patient.diabetes_type = DiabetesType.TYPE1
    patient.insulin_injections_per_day = 3
    # Expected time REMAINING must clear 48 hours, and this patient is already
    # part-way through their admission, so the total has to account for that.
    patient.steps_on_ward = 0
    patient.expected_los_hours = 96.0
    patient.has_capacity = True
    patient.pregnant_or_breastfeeding = False
    patient.end_of_life = False
    patient.will_consent = True
    patient.consent_asked = True
    patient.consent_declined = False
    patient.enrolment = EnrolmentStatus.NOT_ENROLLED

    engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(patient.bed)
    _o, _r, _t, _tr, info = engine.step(int(Action.ENROL))

    assert info["kpi"]["correct_enrolments"] == 1
    assert info["reward_components"]["correct_enrolment"] > 0


def test_administrative_work_during_an_urgent_alarm_is_penalised():
    """Unsafe prioritisation: paperwork while somebody is alarming."""
    from ward_cgm_sim.core.alarms import Alarm, AlarmKind

    engine = WardEngine(SimConfig(), seed=9)
    patient = next(p for p in engine.flow.patients() if p.is_enrolled)
    engine.active_alarms[patient.bed] = Alarm(
        bed=patient.bed,
        kind=AlarmKind.SEVERE_HYPO,
        raised_step=engine.step_index,
        cgm_value=2.5,
    )

    assert Action.PRIORITISE_BEDFLOW in ADMIN_ACTIONS
    _o, _r, _t, _tr, info = engine.step(int(Action.PRIORITISE_BEDFLOW))
    assert info["reward_components"].get("unsafe_prioritisation", 0.0) < 0


def test_invalid_actions_are_penalised_but_do_not_crash():
    engine = WardEngine(SimConfig(), seed=10)
    # Stand in a corridor with no adjacent bed, then attempt a bedside action.
    engine.agent_x, engine.agent_y = engine.ward_map.agent_start
    _o, _r, _t, _tr, info = engine.step(int(Action.POC_GLUCOSE_TEST))
    assert info["kpi"]["invalid_actions"] >= 1


# ---------------------------------------------------------------------------
# Bed flow
# ---------------------------------------------------------------------------

def test_discharge_pipeline_completes_and_frees_the_bed():
    engine = WardEngine(SimConfig(), seed=11)
    patient = next(engine.flow.patients())
    bed = patient.bed
    patient.discharge_stage = DischargeStage.REVIEWED

    engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(bed)
    engine.step(int(Action.SUPPORT_DISCHARGE))
    assert patient.discharge_stage is DischargeStage.SUPPORTED

    for _ in range(20):
        engine.step(int(Action.WAIT))
        if patient.discharge_stage is DischargeStage.DISCHARGED:
            break

    assert patient.discharge_stage is DischargeStage.DISCHARGED
    assert patient in engine.flow.discharged
    assert engine.flow.patient_at_bed(bed) is not patient


def test_discharge_readiness_respects_expected_length_of_stay():
    """A patient documented for a long stay must not evaporate mid-shift.

    Expected length of stay is an enrolment criterion, so discharging such a
    patient an hour later would contradict the basis for enrolling them.
    """
    engine = WardEngine(SimConfig(), seed=12)
    patient = next(engine.flow.patients())
    patient.expected_los_hours = 240.0
    patient.steps_on_ward = 0
    patient.discharge_stage = DischargeStage.NOT_READY

    for _ in range(100):
        engine.flow._step_discharge(patient, engine.step_index, {"became_ready": 0})
    assert patient.discharge_stage is DischargeStage.NOT_READY


# ---------------------------------------------------------------------------
# Regressions found in review
# ---------------------------------------------------------------------------

def test_a_discharged_patient_does_not_silence_the_next_occupant():
    """Alarms are keyed by bed, and discharged patients keep their old bed.

    Iterating the cumulative discharged list every step cleared alarms on that
    bed forever, so whoever was admitted into it next could deteriorate in
    silence. That is the worst possible failure mode for a safety model.
    """
    from ward_cgm_sim.core.alarms import Alarm, AlarmKind

    engine = WardEngine(SimConfig(), seed=13)
    leaving = next(engine.flow.patients())
    bed = leaving.bed
    leaving.discharge_stage = DischargeStage.REVIEWED
    engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(bed)
    engine.step(int(Action.SUPPORT_DISCHARGE))

    for _ in range(25):
        engine.step(int(Action.WAIT))
        if leaving.discharge_stage is DischargeStage.DISCHARGED:
            break
    assert leaving.discharge_stage is DischargeStage.DISCHARGED, (
        "positive control: the first patient must actually leave"
    )

    # Put somebody new in that bed and raise an alarm on them.
    newcomer = engine.flow.patient_at_bed(bed)
    if newcomer is None:
        newcomer = next(p for p in engine.flow.patients() if p.bed != bed)
        engine.flow.beds[bed] = newcomer
        newcomer.bed = bed
    engine.active_alarms[bed] = Alarm(
        bed=bed,
        kind=AlarmKind.SEVERE_HYPO,
        raised_step=engine.step_index,
        cgm_value=2.6,
    )

    engine.step(int(Action.WAIT))
    assert bed in engine.active_alarms, (
        "a previous occupant's discharge silenced the new patient's alarm"
    )


def test_eligibility_uses_expected_remaining_stay_not_total():
    """A patient 47 hours into a 48-hour admission is leaving tomorrow."""
    engine = WardEngine(SimConfig(), seed=14)
    patient = next(engine.flow.patients())
    patient.expected_los_hours = 60.0

    patient.steps_on_ward = 0
    assert patient.expected_los_at_least_48h, "positive control: fresh admission qualifies"

    # 47 hours in: only 13 hours remain, so the criterion must fail.
    patient.steps_on_ward = int(47 * 60 / SimConfig().minutes_per_step)
    assert not patient.expected_los_at_least_48h
    assert patient.expected_remaining_hours == pytest.approx(13.0, abs=0.2)
