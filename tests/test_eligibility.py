"""Eligibility, exclusion and de-enrolment truth table.

Every inclusion criterion and every exclusion named in the study protocol gets
its own case, plus each way a patient can *become* ineligible mid-shift. These
are the rules that decide whether a real patient would have a sensor fitted, so
partial coverage here would be the worst kind of gap.

ACADEMIC MODEL - the criteria encoded here are the simulated study's, not
clinical guidance.
"""

import pytest

from ward_cgm_sim.config import SimConfig
from ward_cgm_sim.core.eligibility import (
    can_enrol,
    evaluate_eligibility,
    hard_exclusions,
    is_eligible_pre_consent,
    should_deenrol,
)
from ward_cgm_sim.core.patient import (
    DiabetesType,
    EnrolmentStatus,
    PatientState,
    Specialty,
)


def make_patient(**overrides) -> PatientState:
    """A patient who satisfies every inclusion criterion, before overrides."""
    defaults = dict(
        patient_id=1,
        bed=0,
        specialty=Specialty.MEDICAL,
        diabetes_type=DiabetesType.TYPE1,
        insulin_injections_per_day=3,
        has_capacity=True,
        will_consent=True,
        expected_los_hours=72.0,
        pregnant_or_breastfeeding=False,
        end_of_life=False,
        hypo_risk=0.5,
        hyper_risk=0.5,
        target_glucose=8.0,
        true_glucose=8.0,
    )
    defaults.update(overrides)
    return PatientState(**defaults)


def consented(patient: PatientState) -> PatientState:
    patient.consent_asked = True
    return patient


# ---------------------------------------------------------------------------
# Inclusion
# ---------------------------------------------------------------------------

def test_fully_eligible_patient_is_eligible():
    """Positive control: the baseline patient must pass, or every negative
    case below would pass vacuously."""
    result = evaluate_eligibility(consented(make_patient()))
    assert result.eligible, result.reasons
    assert result.reasons == ()


@pytest.mark.parametrize(
    "diabetes_type",
    [DiabetesType.TYPE1, DiabetesType.TYPE2, DiabetesType.TYPE3C, DiabetesType.OTHER],
)
def test_all_diabetes_types_are_eligible(diabetes_type):
    """Type 1, type 2, type 3c and other categories all qualify."""
    patient = consented(make_patient(diabetes_type=diabetes_type))
    assert evaluate_eligibility(patient).eligible


@pytest.mark.parametrize("injections", [2, 3, 4])
def test_two_or_more_injections_qualifies(injections):
    patient = consented(make_patient(insulin_injections_per_day=injections))
    assert evaluate_eligibility(patient).eligible


@pytest.mark.parametrize("hours", [48.0, 72.0, 240.0])
def test_stay_of_at_least_48_hours_qualifies(hours):
    patient = consented(make_patient(expected_los_hours=hours))
    assert evaluate_eligibility(patient).eligible


# ---------------------------------------------------------------------------
# Exclusion - one case per criterion in the protocol
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"diabetes_type": DiabetesType.NONE}, "no_diabetes"),
        ({"insulin_injections_per_day": 1}, "fewer_than_two_injections"),
        ({"insulin_injections_per_day": 0}, "fewer_than_two_injections"),
        ({"expected_los_hours": 47.9}, "expected_stay_under_48h"),
        ({"expected_los_hours": 12.0}, "expected_stay_under_48h"),
        ({"has_capacity": False}, "lacks_capacity"),
        ({"pregnant_or_breastfeeding": True}, "pregnant_or_breastfeeding"),
        ({"end_of_life": True}, "end_of_life"),
    ],
)
def test_each_exclusion_criterion_excludes(overrides, expected_reason):
    patient = consented(make_patient(**overrides))
    result = evaluate_eligibility(patient)
    assert not result.eligible
    assert expected_reason in result.reasons, result.reasons


def test_declining_consent_excludes():
    patient = make_patient()
    patient.consent_asked = True
    patient.consent_declined = True
    result = evaluate_eligibility(patient)
    assert not result.eligible
    assert "declined" in result.reasons


def test_never_asked_for_consent_is_not_eligible():
    """Meeting the clinical criteria is not consent."""
    result = evaluate_eligibility(make_patient())
    assert not result.eligible
    assert "consent_not_obtained" in result.reasons


def test_would_not_consent_excludes_even_if_asked():
    patient = consented(make_patient(will_consent=False))
    assert not evaluate_eligibility(patient).eligible


def test_exclusions_accumulate():
    """A patient failing several criteria reports all of them."""
    patient = consented(
        make_patient(
            diabetes_type=DiabetesType.NONE,
            insulin_injections_per_day=0,
            expected_los_hours=6.0,
            has_capacity=False,
        )
    )
    reasons = evaluate_eligibility(patient).reasons
    assert len(reasons) >= 4, reasons
    for expected in (
        "no_diabetes",
        "fewer_than_two_injections",
        "expected_stay_under_48h",
        "lacks_capacity",
    ):
        assert expected in reasons


# ---------------------------------------------------------------------------
# Pre-consent eligibility (drives the "missed an eligible patient" penalty)
# ---------------------------------------------------------------------------

def test_pre_consent_eligibility_requires_willingness():
    assert is_eligible_pre_consent(make_patient())
    assert not is_eligible_pre_consent(make_patient(will_consent=False))


def test_pre_consent_eligibility_respects_hard_exclusions():
    assert not is_eligible_pre_consent(make_patient(end_of_life=True))
    assert not is_eligible_pre_consent(make_patient(expected_los_hours=24.0))


# ---------------------------------------------------------------------------
# De-enrolment - every way eligibility can lapse mid-shift
# ---------------------------------------------------------------------------

def enrolled_patient(**overrides) -> PatientState:
    patient = consented(make_patient(**overrides))
    patient.enrolment = EnrolmentStatus.ENROLLED
    return patient


def test_enrolled_and_still_eligible_needs_no_deenrolment():
    """Positive control for the de-enrolment cases below."""
    needs, reasons = should_deenrol(enrolled_patient())
    assert not needs
    assert reasons == ()


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"insulin_injections_per_day": 1}, "fewer_than_two_injections"),
        ({"end_of_life": True}, "end_of_life"),
        ({"pregnant_or_breastfeeding": True}, "pregnant_or_breastfeeding"),
        ({"expected_los_hours": 24.0}, "expected_stay_under_48h"),
        ({"has_capacity": False}, "lacks_capacity"),
    ],
)
def test_losing_a_criterion_triggers_deenrolment(overrides, expected_reason):
    patient = enrolled_patient()
    for key, value in overrides.items():
        setattr(patient, key, value)
    needs, reasons = should_deenrol(patient)
    assert needs
    assert expected_reason in reasons


def test_withdrawing_consent_triggers_deenrolment():
    patient = enrolled_patient()
    patient.consent_declined = True
    needs, reasons = should_deenrol(patient)
    assert needs
    assert "withdrew_consent" in reasons


def test_deenrolment_check_ignores_patients_who_are_not_enrolled():
    needs, reasons = should_deenrol(make_patient(end_of_life=True))
    assert not needs and reasons == ()


def test_consent_is_not_retested_for_enrolled_patients():
    """An enrolled patient has already consented.

    Re-testing consent here would flag the entire cohort for de-enrolment,
    since ``consent_asked`` says nothing about ongoing agreement.
    """
    patient = enrolled_patient()
    patient.consent_asked = False  # e.g. record lost
    needs, _ = should_deenrol(patient)
    assert not needs


# ---------------------------------------------------------------------------
# Action legality
# ---------------------------------------------------------------------------

def test_cannot_enrol_before_asking_consent():
    ok, reason = can_enrol(make_patient())
    assert not ok and reason == "consent_not_asked"


def test_cannot_enrol_after_a_decline():
    patient = make_patient()
    patient.consent_asked = True
    patient.consent_declined = True
    ok, reason = can_enrol(patient)
    assert not ok and reason == "consent_declined"


def test_cannot_enrol_twice():
    ok, reason = can_enrol(enrolled_patient())
    assert not ok and reason == "already_enrolled"


def test_can_enrol_a_consented_patient():
    ok, reason = can_enrol(consented(make_patient()))
    assert ok and reason == ""


def test_can_enrol_does_not_pre_judge_eligibility():
    """Enrolling an ineligible patient must be *possible* - and penalised.

    If the action were blocked outright the agent could never make the mistake,
    and the reward for avoiding it would be meaningless.
    """
    patient = consented(make_patient(expected_los_hours=6.0))
    ok, _ = can_enrol(patient)
    assert ok
    assert not evaluate_eligibility(patient).eligible


def test_hard_exclusions_exclude_consent_state():
    """hard_exclusions covers clinical criteria only, never consent."""
    patient = make_patient()
    patient.consent_declined = True
    assert hard_exclusions(patient) == ()
