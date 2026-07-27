"""CGM alarm generation, thresholds and the silent-failure behaviour.

The single most important case in this file is that signal loss raises NO
alarm. That is a deliberate model of a real failure mode: the data simply stops
arriving, and somebody has to notice. A test suite that only checked alarms
fire would miss it entirely.

ACADEMIC MODEL - thresholds are the simulated study's, not clinical guidance.
"""

import pytest

from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.alarms import (
    AlarmKind,
    SIGNIFICANT_ALARMS,
    URGENT_ALARMS,
    evaluate_alarms,
    is_false_alarm,
    rate_of_change,
)
from ward_cgm_sim.core.engine import WardEngine
from ward_cgm_sim.core.glucose import step_sensor
from ward_cgm_sim.core.patient import EnrolmentStatus

from test_eligibility import make_patient


def enrolled(**overrides):
    patient = make_patient(**overrides)
    patient.enrolment = EnrolmentStatus.ENROLLED
    return patient


CFG = SimConfig()


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_in_range_value_raises_nothing():
    """Positive control: a normal reading must be silent."""
    assert evaluate_alarms(enrolled(), 7.0, 0, CFG) == []


@pytest.mark.parametrize("value", [3.89, 3.5, 3.01])
def test_below_hypo_threshold_raises_hypo(value):
    assert AlarmKind.HYPO in evaluate_alarms(enrolled(), value, 0, CFG)


def test_exactly_at_hypo_threshold_does_not_alarm():
    """3.9 is the boundary: below alarms, at does not."""
    assert evaluate_alarms(enrolled(), CFG.alarms.hypo_threshold, 0, CFG) == []


@pytest.mark.parametrize("value", [2.99, 2.0, 1.5])
def test_below_severe_threshold_raises_severe_not_plain_hypo(value):
    kinds = evaluate_alarms(enrolled(), value, 0, CFG)
    assert AlarmKind.SEVERE_HYPO in kinds
    assert AlarmKind.HYPO not in kinds, "severe supersedes plain hypo"


@pytest.mark.parametrize("value", [14.1, 20.0])
def test_above_hyper_threshold_raises_hyper(value):
    assert AlarmKind.HYPER in evaluate_alarms(enrolled(), value, 0, CFG)


def test_exactly_at_hyper_threshold_does_not_alarm():
    assert evaluate_alarms(enrolled(), CFG.alarms.hyper_threshold_default, 0, CFG) == []


# ---------------------------------------------------------------------------
# Individualised thresholds - the alarm-fatigue lever
# ---------------------------------------------------------------------------

def test_individualised_threshold_suppresses_nuisance_hyper_alarms():
    """A chronically hyperglycaemic patient can be given a raised threshold.

    Paired with a positive control on the default patient, so this cannot pass
    by the value simply never alarming for anyone.
    """
    value = 16.0
    default_patient = enrolled()
    assert AlarmKind.HYPER in evaluate_alarms(default_patient, value, 0, CFG)

    individualised = enrolled()
    individualised.individualised_hyper_threshold = (
        CFG.alarms.hyper_threshold_individualised
    )
    assert AlarmKind.HYPER not in evaluate_alarms(individualised, value, 0, CFG)


def test_individualised_threshold_still_alarms_when_genuinely_high():
    patient = enrolled()
    patient.individualised_hyper_threshold = CFG.alarms.hyper_threshold_individualised
    assert AlarmKind.HYPER in evaluate_alarms(patient, 19.0, 0, CFG)


def test_individualised_threshold_does_not_affect_hypo_alarms():
    """Raising the hyper threshold must never blunt hypoglycaemia detection."""
    patient = enrolled()
    patient.individualised_hyper_threshold = 18.0
    assert AlarmKind.HYPO in evaluate_alarms(patient, 3.5, 0, CFG)


# ---------------------------------------------------------------------------
# Rate-of-change
# ---------------------------------------------------------------------------

def test_rapid_fall_alarm_fires_on_a_sustained_drop():
    patient = enrolled()
    patient.cgm_history.extend([12.0, 12.0, 10.0, 8.0, 6.5, 6.0])
    kinds = evaluate_alarms(patient, 6.0, 0, CFG)
    assert AlarmKind.RAPID_FALL in kinds


def test_rapid_rise_alarm_fires_on_a_sustained_climb():
    patient = enrolled()
    patient.cgm_history.extend([5.0, 5.0, 7.0, 9.0, 10.5, 11.0])
    assert AlarmKind.RAPID_RISE in evaluate_alarms(patient, 11.0, 0, CFG)


def test_stable_trace_raises_no_rate_alarm():
    """Positive control for the two cases above."""
    patient = enrolled()
    patient.cgm_history.extend([8.0, 8.1, 7.9, 8.0, 8.1, 8.0])
    kinds = evaluate_alarms(patient, 8.0, 0, CFG)
    assert AlarmKind.RAPID_FALL not in kinds
    assert AlarmKind.RAPID_RISE not in kinds


def test_rate_of_change_needs_enough_history():
    patient = enrolled()
    patient.cgm_history.extend([8.0, 7.0])
    assert rate_of_change(patient, CFG.alarms.roc_window_steps) is None


def test_rate_of_change_is_smoothed():
    """A single-sample spike must not by itself produce a trend.

    Differencing raw samples doubles the noise; real devices smooth first.
    """
    patient = enrolled()
    patient.cgm_history.extend([8.0, 8.0, 8.0, 8.0, 8.0, 12.0])
    raw = patient.cgm_history[-1] - patient.cgm_history[-1 - CFG.alarms.roc_window_steps]
    smoothed = rate_of_change(patient, CFG.alarms.roc_window_steps)
    assert abs(smoothed) < abs(raw), "trend should be damped relative to raw difference"


# ---------------------------------------------------------------------------
# Silent failure - the case a naive suite would miss
# ---------------------------------------------------------------------------

def test_signal_loss_raises_no_alarm():
    """Missing data must be silent. The agent has to notice the gap itself."""
    assert evaluate_alarms(enrolled(), None, 0, CFG) == []


def test_unenrolled_patient_raises_no_alarm():
    patient = make_patient()
    assert patient.enrolment is not EnrolmentStatus.ENROLLED
    assert evaluate_alarms(patient, 2.0, 0, CFG) == []


def test_signal_loss_increments_staleness_without_alarming():
    """End-to-end: a lost sensor produces a growing gap and no alarm."""
    import random

    cfg = SimConfig()
    patient = enrolled()
    patient.glucose_history.extend([2.5] * 8)  # dangerously low, but unseen
    patient.signal_lost = True
    patient.signal_loss_steps_left = 5

    rng = random.Random(0)
    for _ in range(4):
        value = step_sensor(patient, rng, cfg)
        assert value is None
        assert evaluate_alarms(patient, value, 0, cfg) == []

    assert patient.steps_since_valid_cgm >= 4, "staleness must be visible to the agent"


def test_engine_does_not_alarm_while_a_patient_is_off_the_ward():
    """A patient in theatre is out of sensor range; a live alarm would strand
    the agent at an empty bed."""
    engine = WardEngine(SimConfig(), seed=2)
    patient = next(p for p in engine.flow.patients() if p.is_enrolled)
    from ward_cgm_sim.core.patient import Location

    patient.location = Location.OFF_WARD
    patient.true_glucose = 2.2
    engine._update_alarms(patient, 2.2)
    assert patient.bed not in engine.active_alarms


# ---------------------------------------------------------------------------
# False-alarm classification
# ---------------------------------------------------------------------------

def test_alarm_on_a_genuinely_low_patient_is_not_false():
    patient = enrolled(true_glucose=3.0)
    assert not is_false_alarm(patient, AlarmKind.HYPO, CFG)


def test_alarm_on_a_clearly_normal_patient_is_false():
    patient = enrolled(true_glucose=8.0)
    assert is_false_alarm(patient, AlarmKind.HYPO, CFG)


def test_borderline_value_is_not_counted_as_a_false_alarm():
    """A patient sitting just under the threshold is not a nuisance alarm.

    Without the margin, noise nudging a genuinely borderline patient across the
    line would be scored as a false alarm - an artefact of the definition
    rather than real alarm burden.
    """
    margin = CFG.alarms.false_alarm_margin
    borderline = enrolled(true_glucose=CFG.alarms.hypo_threshold + margin / 2)
    assert not is_false_alarm(borderline, AlarmKind.HYPO, CFG)


def test_trend_alarm_on_an_out_of_range_patient_is_not_false():
    patient = enrolled(true_glucose=3.2)
    patient.glucose_history.extend([3.2] * 6)
    assert not is_false_alarm(patient, AlarmKind.RAPID_FALL, CFG)


# ---------------------------------------------------------------------------
# Alarm taxonomy
# ---------------------------------------------------------------------------

def test_hypoglycaemia_alarms_are_urgent():
    assert AlarmKind.HYPO in URGENT_ALARMS
    assert AlarmKind.SEVERE_HYPO in URGENT_ALARMS
    assert AlarmKind.RAPID_FALL in URGENT_ALARMS


def test_significant_alarms_require_point_of_care_confirmation():
    for kind in (AlarmKind.HYPO, AlarmKind.SEVERE_HYPO, AlarmKind.HYPER):
        assert kind in SIGNIFICANT_ALARMS


def test_persistence_suppresses_a_single_stray_reading():
    """One out-of-range sample must not alarm when persistence is required."""
    cfg = SimConfig()
    assert cfg.alarms.persistence_readings >= 2

    engine = WardEngine(cfg, seed=3)
    patient = next(p for p in engine.flow.patients() if p.is_enrolled)
    patient.alarm_streak = {}
    engine._update_alarms(patient, 3.0)
    assert patient.bed not in engine.active_alarms, "first reading should not alarm"

    engine._update_alarms(patient, 3.0)
    assert patient.bed in engine.active_alarms, "second consecutive reading should alarm"


def test_persistence_survives_oscillation_across_the_severe_threshold():
    """Readings alternating 2.9 / 3.1 are continuously hypoglycaemic.

    Tracking persistence per exact alarm kind meant the streak reset on every
    reading, so a genuinely and dangerously low patient never alarmed at all.
    """
    cfg = SimConfig()
    assert cfg.alarms.persistence_readings >= 2

    engine = WardEngine(cfg, seed=5)
    patient = next(p for p in engine.flow.patients() if p.is_enrolled)
    patient.alarm_streak = {}

    engine._update_alarms(patient, 2.9)   # severe hypo reading
    assert patient.bed not in engine.active_alarms, "first reading should not alarm"
    engine._update_alarms(patient, 3.1)   # still hypo, different severity
    assert patient.bed in engine.active_alarms, (
        "sustained hypoglycaemia failed to alarm because the severity changed"
    )


def test_persistence_still_suppresses_an_isolated_artefact():
    """Positive counterpart: a single spike between normal readings is silent."""
    cfg = SimConfig()
    engine = WardEngine(cfg, seed=5)
    patient = next(p for p in engine.flow.patients() if p.is_enrolled)
    patient.alarm_streak = {}

    engine._update_alarms(patient, 2.8)   # artefact
    engine._update_alarms(patient, 7.0)   # back to normal - streak must reset
    engine._update_alarms(patient, 2.8)   # another isolated artefact
    assert patient.bed not in engine.active_alarms
