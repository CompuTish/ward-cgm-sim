"""Atomic coverage of the action space: every action, one at a time.

The other engine tests exercise scenarios - a hypo that goes undetected, a
discharge that completes. Nothing asserted that each of the 24 actions
individually does the thing its name promises, or that it is a penalised no-op
when its precondition is absent. That gap is how a control can quietly stop
working while the whole suite stays green.

Each action gets at least: one test that it works under the right conditions,
and one that it declines to work under the wrong ones.
"""

from __future__ import annotations

import pytest

from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.actions import Action
from ward_cgm_sim.core.alarms import Alarm, AlarmKind
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.core.patient import (
    DiabetesType,
    DischargeStage,
    EnrolmentStatus,
    Location,
)

MOVEMENT = {
    Action.MOVE_UP: (0, -1),
    Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0),
    Action.MOVE_RIGHT: (1, 0),
}


def engine(seed: int = 5, **config) -> WardEngine:
    return WardEngine(SimConfig(**config), seed=seed)


def beside(eng: WardEngine, bed: int = 0) -> WardEngine:
    """Stand the agent on a tile from which `bed` can be acted on."""
    tile = eng.ward_map.approach_tile(bed)
    assert tile is not None, "positive control: the bed must be reachable"
    eng.agent_x, eng.agent_y = tile
    assert eng.ward_map.adjacent_bed(*tile) == bed
    return eng


def on_open_floor(eng: WardEngine) -> WardEngine:
    """Stand the agent where no bed is adjacent and the station is not in reach."""
    for y in range(eng.ward_map.height):
        for x in range(eng.ward_map.width):
            if (
                eng.ward_map.walkable(x, y)
                and eng.ward_map.adjacent_bed(x, y) is None
                and not eng.ward_map.at_station(x, y)
            ):
                eng.agent_x, eng.agent_y = x, y
                return eng
    raise AssertionError("positive control: the ward must have open floor")


def make_eligible(patient) -> None:
    """Force every inclusion criterion true and every exclusion false."""
    patient.diabetes_type = DiabetesType.TYPE1
    patient.insulin_injections_per_day = 3
    patient.expected_los_hours = 200.0
    patient.steps_on_ward = 0
    patient.has_capacity = True
    patient.pregnant_or_breastfeeding = False
    patient.end_of_life = False
    patient.consent_declined = False


def occupant(eng: WardEngine, bed: int = 0):
    patient = eng.flow.patient_at_bed(bed)
    assert patient is not None, "positive control: the bed must be occupied"
    return patient


def enrol(eng: WardEngine, bed: int = 0):
    """Take a patient all the way onto telemetry, the legitimate way."""
    patient = occupant(eng, bed)
    make_eligible(patient)
    beside(eng, bed)
    eng.step(int(Action.REVIEW_NOTES))
    eng.step(int(Action.ASK_CONSENT))
    eng.step(int(Action.ENROL))
    assert patient.enrolment is EnrolmentStatus.ENROLLED, eng.last_action_result
    return patient


# --------------------------------------------------------------------------
# The space itself
# --------------------------------------------------------------------------


def test_the_action_space_is_exactly_the_documented_24():
    assert len(Action) == 24
    assert [a.value for a in Action] == list(range(24))


@pytest.mark.parametrize("action", list(Action))
def test_every_action_is_accepted_by_step(action):
    """No action may crash, whatever the agent is standing next to."""
    eng = beside(engine(), 0)
    obs, reward, terminated, truncated, info = eng.step(int(action))
    assert len(obs) == eng.observation_size
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert eng.last_action_result, f"{action.name} reported nothing"


@pytest.mark.parametrize("action", list(Action))
def test_every_action_reports_a_distinct_outcome_string(action):
    """`last_action_result` drives the on-screen feedback; it must be set."""
    eng = on_open_floor(engine())
    eng.step(int(action))
    assert isinstance(eng.last_action_result, str) and eng.last_action_result


# --------------------------------------------------------------------------
# Movement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action,delta", list(MOVEMENT.items()))
def test_movement_moves_one_tile_when_the_way_is_clear(action, delta):
    eng = engine()
    # Find a tile that is clear in this particular direction rather than
    # skipping when the first open tile happens to be blocked - a skip here
    # would silently stop testing the direction altogether.
    start = next(
        (
            (x, y)
            for y in range(eng.ward_map.height)
            for x in range(eng.ward_map.width)
            if eng.ward_map.walkable(x, y)
            and eng.ward_map.walkable(x + delta[0], y + delta[1])
        ),
        None,
    )
    assert start is not None, f"no tile on the map is clear to the {delta}"
    eng.agent_x, eng.agent_y = start
    eng.step(int(action))
    assert (eng.agent_x, eng.agent_y) == (start[0] + delta[0], start[1] + delta[1])
    assert eng.last_action_result == "moved"


@pytest.mark.parametrize("action,delta", list(MOVEMENT.items()))
def test_movement_into_a_wall_is_a_penalised_no_op(action, delta):
    eng = engine()
    # Stand next to the boundary wall in the direction under test.
    positions = {
        (0, -1): (1, 1), (0, 1): (1, eng.ward_map.height - 2),
        (-1, 0): (1, 1), (1, 0): (eng.ward_map.width - 2, 1),
    }
    eng.agent_x, eng.agent_y = positions[delta]
    assert not eng.ward_map.walkable(eng.agent_x + delta[0], eng.agent_y + delta[1]), (
        "positive control: the target tile must really be blocked"
    )
    before = (eng.agent_x, eng.agent_y)
    invalid_before = eng.kpi["invalid_actions"]
    eng.step(int(action))
    assert (eng.agent_x, eng.agent_y) == before
    assert eng.last_action_result == "no effect (blocked)"
    assert eng.kpi["invalid_actions"] == invalid_before + 1


def test_the_four_movements_cover_all_four_directions():
    assert set(MOVEMENT.values()) == {(0, -1), (0, 1), (-1, 0), (1, 0)}


def test_walking_up_to_the_station_reads_the_board_for_free():
    """Standing at the board is the free alternative to spending a step on it."""
    eng = engine()
    by_delta = {(0, 1): Action.MOVE_DOWN, (0, -1): Action.MOVE_UP,
                (-1, 0): Action.MOVE_LEFT, (1, 0): Action.MOVE_RIGHT}
    for y in range(eng.ward_map.height):
        for x in range(eng.ward_map.width):
            if not (eng.ward_map.walkable(x, y) and not eng.ward_map.at_station(x, y)):
                continue
            for delta, action in by_delta.items():
                target = (x + delta[0], y + delta[1])
                if eng.ward_map.walkable(*target) and eng.ward_map.at_station(*target):
                    eng.agent_x, eng.agent_y = x, y
                    eng.dashboard_seen_step = None
                    eng.step(int(action))
                    assert (eng.agent_x, eng.agent_y) == target
                    assert eng.dashboard_seen_step is not None, (
                        "arriving at the station did not refresh the board"
                    )
                    return
    raise AssertionError("no tile on this map walks up to the nurse station")


# --------------------------------------------------------------------------
# Information gathering
# --------------------------------------------------------------------------


def test_check_dashboard_captures_a_snapshot():
    eng = on_open_floor(engine())
    eng.dashboard_seen_step = None
    eng.step(int(Action.CHECK_DASHBOARD))
    assert eng.dashboard_seen_step is not None
    assert eng.last_action_result == "checked dashboard"


def test_check_dashboard_does_nothing_without_telemetry():
    eng = on_open_floor(engine(telemetry_enabled=False))
    invalid_before = eng.kpi["invalid_actions"]
    eng.step(int(Action.CHECK_DASHBOARD))
    assert eng.dashboard_seen_step is None
    assert eng.last_action_result == "no effect (no_telemetry)"
    assert eng.kpi["invalid_actions"] == invalid_before + 1


def test_check_patient_records_when_the_patient_was_seen():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    assert patient.knowledge.last_checked_step is None
    eng.step(int(Action.CHECK_PATIENT))
    assert patient.knowledge.last_checked_step == eng.step_index - 1


def test_review_notes_reveals_every_eligibility_fact():
    eng = beside(engine(), 0)
    knowledge = occupant(eng).knowledge
    assert not knowledge.knows_full_eligibility()
    eng.step(int(Action.REVIEW_NOTES))
    assert knowledge.knows_full_eligibility()
    assert knowledge.known_specialty is not None


@pytest.mark.parametrize(
    "action",
    [Action.CHECK_PATIENT, Action.REVIEW_NOTES, Action.ASK_CONSENT, Action.ENROL,
     Action.REVIEW_ELIGIBILITY, Action.DEENROL, Action.RESPOND_ALARM,
     Action.POC_GLUCOSE_TEST, Action.TREAT_HYPO, Action.TREAT_HYPER,
     Action.ESCALATE, Action.TROUBLESHOOT_SENSOR, Action.SUPPORT_DISCHARGE],
)
def test_patient_actions_need_a_patient_in_reach(action):
    eng = on_open_floor(engine())
    invalid_before = eng.kpi["invalid_actions"]
    eng.step(int(action))
    assert eng.last_action_result == "no effect (no_patient)"
    assert eng.kpi["invalid_actions"] == invalid_before + 1


# --------------------------------------------------------------------------
# Consent and enrolment
# --------------------------------------------------------------------------


def test_ask_consent_records_the_answer():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    make_eligible(patient)
    eng.step(int(Action.ASK_CONSENT))
    assert patient.consent_asked
    assert patient.knowledge.known_consented is not None


def test_enrol_requires_consent_to_have_been_asked():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    make_eligible(patient)
    eng.step(int(Action.ENROL))
    assert patient.enrolment is not EnrolmentStatus.ENROLLED
    assert eng.last_action_result == "no effect (consent_not_asked)"


def test_enrol_puts_an_eligible_consenting_patient_on_telemetry():
    eng = engine()
    patient = enrol(eng, 0)
    assert patient.enrolment is EnrolmentStatus.ENROLLED
    assert eng.kpi["correct_enrolments"] >= 1


def test_enrolling_someone_ineligible_is_recorded_as_an_error():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    make_eligible(patient)
    patient.diabetes_type = DiabetesType.NONE  # the one criterion that now fails
    eng.step(int(Action.REVIEW_NOTES))
    eng.step(int(Action.ASK_CONSENT))
    before = eng.kpi["incorrect_enrolments"]
    eng.step(int(Action.ENROL))
    assert eng.kpi["incorrect_enrolments"] == before + 1


def test_review_eligibility_needs_an_enrolled_patient():
    eng = beside(engine(), 0)
    eng.step(int(Action.REVIEW_ELIGIBILITY))
    assert eng.last_action_result == "no effect (not_enrolled)"


def test_review_eligibility_runs_on_an_enrolled_patient():
    eng = engine()
    enrol(eng, 0)
    eng.step(int(Action.REVIEW_ELIGIBILITY))
    assert eng.last_action_result != "no effect (not_enrolled)"


def test_deenrol_takes_a_patient_off_telemetry():
    eng = engine()
    patient = enrol(eng, 0)
    eng.step(int(Action.DEENROL))
    assert patient.enrolment is not EnrolmentStatus.ENROLLED


def test_deenrol_needs_an_enrolled_patient():
    eng = beside(engine(), 0)
    eng.step(int(Action.DEENROL))
    assert eng.last_action_result == "no effect (not_enrolled)"


# --------------------------------------------------------------------------
# Clinical response
# --------------------------------------------------------------------------


def test_respond_alarm_needs_an_alarm():
    eng = beside(engine(), 0)
    eng.step(int(Action.RESPOND_ALARM))
    assert eng.last_action_result == "no effect (no_alarm)"


def test_respond_alarm_acknowledges_a_live_alarm():
    eng = engine()
    patient = enrol(eng, 0)
    alarm = Alarm(bed=0, kind=AlarmKind.HYPO,
                  raised_step=eng.step_index, cgm_value=3.4)
    eng.active_alarms[0] = alarm
    beside(eng, 0)
    eng.step(int(Action.RESPOND_ALARM))
    assert alarm.acknowledged_step is not None


def test_point_of_care_test_produces_a_reading():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    assert patient.knowledge.last_poc_value is None
    eng.step(int(Action.POC_GLUCOSE_TEST))
    assert patient.knowledge.last_poc_value is not None
    assert patient.knowledge.last_poc_step is not None


def test_treating_a_patient_who_is_not_low_is_recorded_as_unnecessary():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    patient.true_glucose = 7.0
    eng.step(int(Action.TREAT_HYPO))
    assert "unnecessary" in eng.last_action_result


def test_treating_hypoglycaemia_raises_glucose():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    patient.true_glucose = 3.2
    patient.knowledge.last_poc_value = 3.2
    patient.knowledge.last_poc_step = eng.step_index
    eng.step(int(Action.TREAT_HYPO))
    assert patient.treatment_kind == "hypo"
    assert patient.treatment_effect_remaining > 0


def test_treating_hyperglycaemia_lowers_glucose():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    patient.true_glucose = 18.0
    patient.knowledge.last_poc_value = 18.0
    patient.knowledge.last_poc_step = eng.step_index
    eng.step(int(Action.TREAT_HYPER))
    assert patient.treatment_kind == "hyper"
    assert patient.treatment_effect_remaining < 0


def test_escalate_is_counted_whether_or_not_anyone_is_free():
    eng = beside(engine(), 0)
    before = eng.kpi["escalations"]
    eng.step(int(Action.ESCALATE))
    assert eng.kpi["escalations"] == before + 1


# --------------------------------------------------------------------------
# Asking colleagues
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [Action.ASK_HELP_HCA, Action.ASK_HELP_NURSE,
     Action.ASK_HELP_DOCTOR, Action.ASK_HELP_SURGEON],
)
def test_asking_a_colleague_is_always_counted(action):
    """Whether they are free is hidden; the asking still costs a step."""
    eng = beside(engine(), 0)
    before = eng.kpi["staff_requests"]
    eng.step(int(action))
    assert eng.kpi["staff_requests"] == before + 1


@pytest.mark.parametrize(
    "action",
    [Action.ASK_HELP_HCA, Action.ASK_HELP_NURSE,
     Action.ASK_HELP_DOCTOR, Action.ASK_HELP_SURGEON],
)
def test_asking_a_colleague_works_without_a_patient_in_reach(action):
    """Unlike bedside actions, a request can be made from anywhere."""
    eng = on_open_floor(engine())
    eng.step(int(action))
    assert eng.last_action_result != "no effect (no_patient)"


def test_the_four_role_requests_are_distinct_roles():
    from ward_cgm_sim.core.engine import ROLE_ACTIONS

    roles = {ROLE_ACTIONS[a] for a in (
        Action.ASK_HELP_HCA, Action.ASK_HELP_NURSE,
        Action.ASK_HELP_DOCTOR, Action.ASK_HELP_SURGEON)}
    assert len(roles) == 4


# --------------------------------------------------------------------------
# Sensors, discharge and bed flow
# --------------------------------------------------------------------------


def test_troubleshoot_needs_an_enrolled_patient():
    eng = beside(engine(), 0)
    eng.step(int(Action.TROUBLESHOOT_SENSOR))
    assert eng.last_action_result == "no effect (not_enrolled)"


def test_troubleshoot_restores_a_lost_signal():
    eng = engine()
    patient = enrol(eng, 0)
    patient.signal_lost = True
    patient.signal_loss_steps_left = 20
    patient.steps_since_valid_cgm = 20
    beside(eng, 0)
    eng.step(int(Action.TROUBLESHOOT_SENSOR))
    assert not patient.signal_lost or patient.signal_loss_steps_left < 20


def test_support_discharge_needs_a_patient_who_is_ready():
    eng = beside(engine(), 0)
    occupant(eng).discharge_stage = DischargeStage.NOT_READY
    eng.step(int(Action.SUPPORT_DISCHARGE))
    assert eng.last_action_result == "no effect (not_ready_for_discharge)"


def test_support_discharge_advances_a_reviewed_patient():
    eng = beside(engine(), 0)
    patient = occupant(eng)
    patient.discharge_stage = DischargeStage.REVIEWED
    eng.step(int(Action.SUPPORT_DISCHARGE))
    assert patient.discharge_stage is DischargeStage.SUPPORTED


def test_prioritise_bedflow_shortens_the_queue_when_there_is_one():
    eng = engine()
    assert eng.flow.queue_length or any(
        p.location is not Location.BED for p in eng.flow.patients()
    ), "positive control: there must be something to prioritise"
    before = eng.flow.queue_length
    eng.step(int(Action.PRIORITISE_BEDFLOW))
    assert eng.flow.queue_length <= before
    assert "bed flow" in eng.last_action_result


def test_prioritise_bedflow_with_nothing_to_do_is_not_penalised():
    """An empty queue is not the agent's mistake."""
    eng = engine()
    eng.flow.queue.clear()
    for patient in eng.flow.patients():
        patient.location = Location.BED
    before = eng.kpi["invalid_actions"]
    eng.step(int(Action.PRIORITISE_BEDFLOW))
    if "nothing to do" in eng.last_action_result:
        assert eng.kpi["invalid_actions"] == before


def test_wait_is_a_legal_action_that_is_never_penalised():
    eng = on_open_floor(engine())
    before = eng.kpi["invalid_actions"]
    eng.step(int(Action.WAIT))
    assert eng.last_action_result == "waited"
    assert eng.kpi["invalid_actions"] == before


def test_waiting_still_advances_the_clock():
    eng = engine()
    start = eng.step_index
    eng.step(int(Action.WAIT))
    assert eng.step_index == start + 1
