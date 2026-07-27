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


def test_an_unread_alarm_is_not_counted_as_detection():
    """"Detection" must mean somebody knows, not that a device fired.

    An alarm sitting on a board nobody has looked at is not detection. Counting
    it would report device latency while claiming to report the time before
    anybody knew - which was the headline number.
    """
    cfg = SimConfig()
    cfg.usual_care.routine_detection_prob = 0.0  # isolate the telemetry route
    # Deterministic low reading so an alarm is guaranteed to be raised.
    cfg.glucose.reversion_rate = 0.0
    cfg.glucose.process_noise = 0.0
    cfg.glucose.meal_prob = 0.0
    cfg.glucose.insulin_prob = 0.0
    cfg.glucose.cgm_noise_sd = 0.0
    cfg.glucose.cgm_bias_sd = 0.0
    cfg.glucose.cgm_spike_prob = 0.0
    engine = WardEngine(cfg, seed=21)
    patient = next(p for p in engine.flow.patients() if p.is_enrolled)

    engine.agent_x, engine.agent_y = 1, 1  # far from the station
    engine.dashboard_seen_step = None
    engine.dashboard_snapshot = {}
    patient.true_glucose = 3.2
    patient.target_glucose = 3.2
    patient.glucose_history.extend([3.2] * 6)

    for _ in range(10):
        engine.step(int(Action.WAIT))

    # POSITIVE CONTROL: an alarm must genuinely have been raised, or this test
    # proves nothing - "nothing was detected" is trivially true if the device
    # never fired in the first place.
    assert engine.kpi["alarms_raised"] > 0, (
        "positive control failed: no alarm was ever raised"
    )
    assert engine.visible_alarms() == [], "the board must remain unread"
    assert engine.kpi["hypo_detected_by_telemetry"] == 0, (
        "an alarm nobody has seen was counted as a detection"
    )


def test_checking_a_patient_can_discover_hypoglycaemia():
    """Bedside checking is a real discovery route, and the only one without
    telemetry. If it does nothing, the counterfactual arm is monitoring-free
    rather than routine-monitoring."""
    cfg = SimConfig()
    cfg.telemetry_enabled = False
    cfg.usual_care.routine_detection_prob = 0.0  # isolate the bedside route
    # Pin the physiology so the patient stays low for the whole test.
    cfg.glucose.reversion_rate = 0.0
    cfg.glucose.process_noise = 0.0
    cfg.glucose.meal_prob = 0.0
    cfg.glucose.insulin_prob = 0.0
    cfg.glucose.hypo_episode_prob = 0.0
    cfg.glucose.hyper_episode_prob = 0.0
    engine = WardEngine(cfg, seed=22)

    patient = next(engine.flow.patients())
    patient.true_glucose = 2.5  # severe: taken to be obviously unwell
    patient.target_glucose = 2.5
    patient.glucose_history.extend([2.5] * 6)
    # Enough steps for the episode to meet the 15-minute event definition.
    for _ in range(4):
        engine.step(int(Action.WAIT))

    engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(patient.bed)
    before = engine.kpi["hypo_detected_by_usual_care"]
    engine.step(int(Action.CHECK_PATIENT))

    assert engine.kpi["hypo_detected_by_usual_care"] > before, (
        "checking a visibly hypoglycaemic patient discovered nothing"
    )


def test_one_episode_is_not_split_by_a_single_recovered_reading():
    """3.5, 3.5, 4.0, 3.5 is one episode, not two."""
    cfg = SimConfig()
    cfg.usual_care.routine_detection_prob = 0.0
    engine = WardEngine(cfg, seed=23)
    patient = next(engine.flow.patients())

    trace = [3.5, 3.5, 3.5, 4.0, 3.5, 3.5, 3.5]
    for value in trace:
        patient.true_glucose = value
        engine._track_hypo_episode(patient)
        engine.step_index += 1

    assert engine.kpi["hypo_episodes"] == 1, (
        f"a brief recovery split one episode into "
        f"{engine.kpi['hypo_episodes']}"
    )


def test_sustained_recovery_does_end_an_episode():
    """Positive counterpart: a real recovery must close the episode."""
    cfg = SimConfig()
    cfg.usual_care.routine_detection_prob = 0.0
    engine = WardEngine(cfg, seed=23)
    patient = next(engine.flow.patients())

    for value in [3.5, 3.5, 3.5] + [6.0] * 6 + [3.5, 3.5, 3.5]:
        patient.true_glucose = value
        engine._track_hypo_episode(patient)
        engine.step_index += 1

    assert engine.kpi["hypo_episodes"] == 2, "sustained recovery should close it"


def test_acknowledging_an_alarm_without_confirming_it_is_penalised():
    """Silencing the board is not responding to it.

    The agent acknowledges every alarm the moment it appears but never runs a
    point-of-care test. That must still accrue a penalty, or the cheapest
    policy is to clear the board and walk away.
    """
    cfg = SimConfig()
    # Hold the patient genuinely low so alarms keep being raised.
    cfg.glucose.reversion_rate = 0.0
    cfg.glucose.process_noise = 0.0
    cfg.glucose.meal_prob = 0.0
    cfg.glucose.insulin_prob = 0.0
    cfg.usual_care.routine_detection_prob = 0.0
    # Silence the sensor noise too, so the alarm cannot flicker back in range
    # and auto-resolve before the deadline is reached.
    cfg.glucose.cgm_noise_sd = 0.0
    cfg.glucose.cgm_bias_sd = 0.0
    cfg.glucose.cgm_spike_prob = 0.0
    engine = WardEngine(cfg, seed=24)

    patient = next(p for p in engine.flow.patients() if p.is_enrolled)
    patient.true_glucose = 3.3
    patient.target_glucose = 3.3
    patient.glucose_history.extend([3.3] * 6)

    raised_any = False
    for _ in range(30):
        # Acknowledge whatever is on the board, but never confirm it.
        for alarm in engine.active_alarms.values():
            if alarm.acknowledged_step is None:
                alarm.acknowledged_step = engine.step_index
                raised_any = True
        _o, _r, terminated, truncated, info = engine.step(int(Action.WAIT))
        if terminated or truncated:
            break

    assert raised_any, "positive control: an alarm must actually have been raised"
    assert info["kpi"]["unconfirmed_alarm_steps"] > 0, (
        "an acknowledged but unconfirmed alarm accrued no penalty"
    )
    assert info["reward_components"].get("unconfirmed_significant_alarm", 0.0) < 0


def test_an_alarm_raised_after_the_board_was_read_is_not_visible():
    """Actions resolve before alarms are generated within a step.

    Comparing raised_step <= dashboard_seen_step therefore let an alarm created
    later in the same step count as already seen - the agent appearing to know
    about something before it existed.
    """
    from ward_cgm_sim.core.alarms import Alarm, AlarmKind

    engine = WardEngine(SimConfig(), seed=25)
    engine.agent_x, engine.agent_y = 1, 1  # away from the station
    engine._read_dashboard()
    assert engine.dashboard_seen_step is not None, "positive control: board read"

    patient = next(p for p in engine.flow.patients() if p.is_enrolled)
    later = Alarm(
        bed=patient.bed,
        kind=AlarmKind.SEVERE_HYPO,
        raised_step=engine.step_index,  # same step as the read
        cgm_value=2.4,
    )
    engine.active_alarms[patient.bed] = later

    assert later not in engine.visible_alarms(), (
        "an alarm raised after the board was read was treated as seen"
    )

    # Positive counterpart: reading again does reveal it.
    engine._read_dashboard()
    assert later in engine.visible_alarms()


def test_interrupted_lows_do_not_qualify_as_a_sustained_event():
    """3.5, 4.0, 4.0, 3.5 is two isolated lows, not fifteen minutes below."""
    cfg = SimConfig()
    cfg.usual_care.routine_detection_prob = 0.0
    engine = WardEngine(cfg, seed=26)
    patient = next(engine.flow.patients())

    for value in [3.5, 4.0, 4.0, 3.5]:
        patient.true_glucose = value
        engine._track_hypo_episode(patient)
        engine.step_index += 1

    assert engine.kpi["hypo_episodes"] == 0, (
        "interrupted lows were counted as a sustained 15-minute event"
    )


def test_three_consecutive_lows_do_qualify():
    """Positive counterpart to the interrupted case."""
    cfg = SimConfig()
    cfg.usual_care.routine_detection_prob = 0.0
    engine = WardEngine(cfg, seed=26)
    patient = next(engine.flow.patients())

    for value in [3.5, 3.5, 3.5]:
        patient.true_glucose = value
        engine._track_hypo_episode(patient)
        engine.step_index += 1

    assert engine.kpi["hypo_episodes"] == 1


def test_bedside_checking_does_not_disturb_the_routine_monitoring_stream():
    """Symptom recognition is an ACTION consequence, not exogenous care.

    Drawing it from rng_care would let the agent's choice to look shift the
    routine-monitoring sequence, desynchronising the two arms.
    """
    def run(check: bool):
        cfg = SimConfig()
        # Routine monitoring off, so any rng_care movement can only come from
        # the bedside check itself.
        cfg.usual_care.routine_detection_prob = 0.0
        engine = WardEngine(cfg, seed=27)
        patient = next(engine.flow.patients())
        # The patient MUST be hypoglycaemic, or no symptom-recognition draw
        # happens at all and this test passes vacuously. Kept just above the
        # severe threshold so the probabilistic branch is the one exercised.
        patient.true_glucose = 3.5
        patient.target_glucose = 3.5
        engine.agent_x, engine.agent_y = engine.ward_map.approach_tile(patient.bed)
        before_action = patient.rng_action.getstate()
        engine.step(int(Action.CHECK_PATIENT if check else Action.WAIT))
        moved_action = patient.rng_action.getstate() != before_action
        return patient.rng_care.getstate(), moved_action

    care_wait, moved_wait = run(False)
    care_check, moved_check = run(True)

    # POSITIVE CONTROL: the check must actually have drawn something, or
    # "rng_care did not move" is trivially true.
    assert moved_check and not moved_wait, (
        "positive control failed: CHECK_PATIENT did not consume an action draw"
    )
    assert care_wait == care_check, (
        "checking a patient consumed the exogenous routine-monitoring stream"
    )


def test_an_aborted_dip_does_not_backdate_a_later_event():
    """3.5, 4.0, 3.5, 3.5, 3.5 must date from step 2, not step 0.

    Carrying the candidate onset through a recovered dip would measure
    detection delay from a low that had already resolved, inflating it.
    """
    cfg = SimConfig()
    cfg.usual_care.routine_detection_prob = 0.0
    engine = WardEngine(cfg, seed=28)
    patient = next(engine.flow.patients())

    for value in [3.5, 4.0, 3.5, 3.5, 3.5]:
        patient.true_glucose = value
        engine._track_hypo_episode(patient)
        engine.step_index += 1

    assert engine.kpi["hypo_episodes"] == 1, (
        "positive control: the final run of three lows must qualify"
    )
    assert patient.hypo_episode_started_step == 2, (
        f"event was dated from step {patient.hypo_episode_started_step}, but the "
        f"qualifying run began at step 2"
    )


def test_an_aborted_dip_does_not_carry_its_detection_forward():
    """A detection banked during a dip that resolved must not count later."""
    cfg = SimConfig()
    cfg.usual_care.routine_detection_prob = 0.0
    engine = WardEngine(cfg, seed=29)
    patient = next(engine.flow.patients())

    patient.true_glucose = 3.5
    engine._track_hypo_episode(patient)
    engine._record_hypo_detection(patient, "usual_care")
    assert patient.hypo_episode_detected, "positive control: detection was banked"

    engine.step_index += 1
    patient.true_glucose = 5.0  # dip resolves before qualifying
    engine._track_hypo_episode(patient)

    assert not patient.hypo_episode_detected, (
        "a detection from an aborted dip survived into the next episode"
    )
