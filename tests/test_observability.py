"""Regression tests for the observability boundary and comparator fairness.

Each test here targets a specific defect found in review: information leaking
into the observation that the agent should have had to work for, and the
rule-based comparator reading hidden state it could not really see. Both are
the kind of bug that makes results look good for the wrong reason.
"""

import pytest

from ward_cgm_sim.agents.rule_based import RuleBasedAgent
from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.actions import Action
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.core.observations import BED_FEATURES, UNKNOWN, WARD_FEATURES
from ward_cgm_sim.core.patient import DischargeStage

CGM_INDEX = 5
STALENESS_INDEX = 6


def bed_feature(engine, bed, index):
    return engine.observation()[WARD_FEATURES + bed * BED_FEATURES + index]


def enrolled_bed(engine):
    return next(p.bed for p in engine.flow.patients() if p.is_enrolled)


def walk_to_station(engine, limit=60):
    """Walk the agent to the nurse station using the map's own pathfinder."""
    target = engine.ward_map.station_tiles[0]
    beside = (target[0] - 1, target[1])
    for _ in range(limit):
        if engine.ward_map.at_station(engine.agent_x, engine.agent_y):
            return True
        start = (engine.agent_x, engine.agent_y)
        nxt = engine.ward_map.next_step_toward(start, beside)
        if nxt is None:
            return False
        dx, dy = nxt[0] - start[0], nxt[1] - start[1]
        action = {
            (1, 0): Action.MOVE_RIGHT,
            (-1, 0): Action.MOVE_LEFT,
            (0, 1): Action.MOVE_DOWN,
            (0, -1): Action.MOVE_UP,
        }[(dx, dy)]
        engine.step(int(action))
    return False


# ---------------------------------------------------------------------------
# Glucose must come from the dashboard, not from a live feed
# ---------------------------------------------------------------------------

def test_glucose_is_unknown_until_the_dashboard_has_been_read():
    """The core leak: CGM values were previously visible ward-wide.

    An agent standing in a corner having never looked at the board must have no
    glucose information at all.
    """
    engine = WardEngine(SimConfig(), seed=3)
    bed = enrolled_bed(engine)
    assert engine.dashboard_seen_step is None, "positive control: board unread"

    assert bed_feature(engine, bed, CGM_INDEX) == UNKNOWN
    assert bed_feature(engine, bed, STALENESS_INDEX) == UNKNOWN


def test_reading_the_dashboard_reveals_glucose():
    """Positive counterpart: the value does appear once the board is read."""
    engine = WardEngine(SimConfig(), seed=3)
    bed = enrolled_bed(engine)
    assert walk_to_station(engine), "positive control: agent reached the station"
    engine.step(int(Action.CHECK_DASHBOARD))

    assert engine.dashboard_seen_step is not None
    assert bed_feature(engine, bed, CGM_INDEX) != UNKNOWN


def test_checking_the_dashboard_costs_a_step_from_anywhere():
    """Telemetry reaches a handheld, so the check works away from the station.

    The cost is the step itself, not the walk - and what it yields is a
    snapshot that then ages, which is what keeps the problem partially
    observable.
    """
    engine = WardEngine(SimConfig(), seed=4)
    engine.agent_x, engine.agent_y = 1, 1
    assert not engine.ward_map.at_station(engine.agent_x, engine.agent_y)
    before = engine.step_index

    engine.step(int(Action.CHECK_DASHBOARD))
    assert engine.dashboard_seen_step == before, "the check should have registered"
    assert engine.step_index == before + 1, "and it should have consumed a step"


def test_standing_at_the_station_refreshes_the_board_for_free():
    engine = WardEngine(SimConfig(), seed=4)
    assert walk_to_station(engine), "positive control: agent reached the station"
    # Arriving at the station reads the board without spending a separate action.
    assert engine.dashboard_seen_step is not None


def test_glucose_information_goes_stale_after_walking_away():
    """Staleness must grow with time since the board was last read."""
    engine = WardEngine(SimConfig(), seed=3)
    bed = enrolled_bed(engine)
    assert walk_to_station(engine)
    engine.step(int(Action.CHECK_DASHBOARD))
    fresh = bed_feature(engine, bed, STALENESS_INDEX)

    # Walk away and wait; the snapshot ages even though the sensor is fine.
    for _ in range(3):
        engine.step(int(Action.MOVE_DOWN))
    for _ in range(8):
        engine.step(int(Action.WAIT))

    stale = bed_feature(engine, bed, STALENESS_INDEX)
    assert stale > fresh, "the agent's picture of the board must age"


def test_no_hidden_patient_field_appears_in_the_observation():
    """Sweep: no unreviewed patient's eligibility or consent should be visible."""
    engine = WardEngine(SimConfig(), seed=6)
    checked = 0
    for patient in engine.flow.patients():
        if patient.knowledge.eligibility_reviewed_step is not None:
            continue
        checked += 1
        assert bed_feature(engine, patient.bed, 3) == UNKNOWN, "eligibility leaked"
        if patient.knowledge.consent_asked_step is None:
            assert bed_feature(engine, patient.bed, 4) == UNKNOWN, "consent leaked"
    assert checked > 15, "positive control: plenty of unreviewed patients to check"


# ---------------------------------------------------------------------------
# The rule-based policy must be a fair comparator, not an oracle
# ---------------------------------------------------------------------------

def test_rule_based_policy_does_not_read_hidden_state():
    """Static check that the comparator only consults observable surfaces.

    A policy that peeks at true discharge stage or per-role staff availability
    would beat any learned policy for reasons that have nothing to do with the
    intervention being studied.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "ward_cgm_sim"
        / "agents"
        / "rule_based.py"
    ).read_text()

    forbidden = [
        "staff.is_available",  # per-role availability is hidden until asked
        ".discharge_stage",  # true stage; the agent knows only what it reviewed
        "steps_since_valid_cgm",  # true sensor state, not the board snapshot
        "individualised_hyper_threshold",  # a personal threshold is not visible
        "hyper_threshold(",  # ditto, via the accessor
        "true_glucose",
    ]
    offenders = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        for name in forbidden:
            if name in stripped:
                offenders.append(f"{name}: {stripped}")
    assert not offenders, "rule-based policy reads hidden state:\n" + "\n".join(offenders)


def test_rule_based_policy_still_completes_a_shift():
    """Positive control for the static check above.

    Without this, deleting the policy's body entirely would satisfy the
    forbidden-substring test.
    """
    engine = WardEngine(SimConfig(), seed=8)
    agent = RuleBasedAgent()
    agent.reset()
    steps = 0
    while True:
        _o, _r, terminated, truncated, info = engine.step(agent.act(engine))
        steps += 1
        if terminated or truncated:
            break
    assert steps > 100, "policy should survive most of a shift"
    assert info["kpi"]["alarms_acknowledged"] >= 0


def test_rule_based_policy_reads_the_dashboard():
    """It must actually go and look, since glucose is no longer free."""
    engine = WardEngine(SimConfig(), seed=9)
    agent = RuleBasedAgent()
    agent.reset()
    for _ in range(80):
        _o, _r, t, tr, _i = engine.step(agent.act(engine))
        if t or tr:
            break
    assert engine.dashboard_seen_step is not None, (
        "the policy never read the telemetry board, so it is flying blind"
    )
