"""Eligibility, enrolment and de-enrolment logic for CGM telemetry.

Inclusion (ALL must hold):
  * diabetes of any type (type 1, type 2, type 3c/other)
  * prescribed two or more insulin injections per day
  * expected to remain on the ward for at least 48 hours
  * has capacity to give verbal informed consent
  * gives verbal informed consent

Exclusion (ANY excludes):
  * fewer than two insulin injections per day
  * expected ward stay under 48 hours
  * lacks capacity to consent
  * declines CGM telemetry
  * pregnancy or breastfeeding
  * end-of-life care
  * a later change in eligibility, e.g. insulin reduced to once daily

ACADEMIC MODEL ONLY - simplified placeholders, not clinical guidance.
Stdlib only (ships to the browser build).
"""

from dataclasses import dataclass

from .patient import EnrolmentStatus, PatientState


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reasons: tuple[str, ...]  # exclusion reasons; empty when eligible

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.eligible


def hard_exclusions(patient: PatientState) -> tuple[str, ...]:
    """Exclusions that do not depend on consent having been asked."""
    reasons: list[str] = []
    if not patient.has_diabetes:
        reasons.append("no_diabetes")
    if not patient.two_or_more_injections:
        reasons.append("fewer_than_two_injections")
    if not patient.expected_los_at_least_48h:
        reasons.append("expected_stay_under_48h")
    if not patient.has_capacity:
        reasons.append("lacks_capacity")
    if patient.pregnant_or_breastfeeding:
        reasons.append("pregnant_or_breastfeeding")
    if patient.end_of_life:
        reasons.append("end_of_life")
    return tuple(reasons)


def evaluate_eligibility(patient: PatientState, require_consent: bool = True) -> EligibilityResult:
    """Ground-truth eligibility. Hidden from the agent."""
    reasons = list(hard_exclusions(patient))
    if require_consent:
        if patient.consent_declined:
            reasons.append("declined")
        elif not patient.consent_asked:
            reasons.append("consent_not_obtained")
        elif not patient.will_consent:
            reasons.append("declined")
    return EligibilityResult(eligible=not reasons, reasons=tuple(reasons))


def is_eligible_pre_consent(patient: PatientState) -> bool:
    """Would this patient be eligible if they consented?

    Used for the "missed an eligible patient" penalty: a patient who would have
    said yes but was never asked still counts as missed.
    """
    if hard_exclusions(patient):
        return False
    return patient.will_consent


def should_deenrol(patient: PatientState) -> tuple[bool, tuple[str, ...]]:
    """Whether an enrolled patient no longer meets the criteria.

    Consent is deliberately excluded from this check: an enrolled patient has
    already consented, so re-testing consent here would flag everyone. What
    matters is a *change* - insulin reduced to once daily, a revised discharge
    plan taking the stay under 48 hours, transition to end-of-life care, or a
    newly documented pregnancy.
    """
    if not patient.is_enrolled:
        return False, ()
    reasons = hard_exclusions(patient)
    if patient.consent_declined:
        reasons = reasons + ("withdrew_consent",)
    return bool(reasons), reasons


def can_enrol(patient: PatientState) -> tuple[bool, str]:
    """Whether ENROL is a valid action right now, ignoring eligibility truth.

    The agent is allowed to attempt enrolment on any consented, not-currently-
    enrolled patient; whether that was *correct* is settled by the reward.
    """
    if patient.enrolment is EnrolmentStatus.ENROLLED:
        return False, "already_enrolled"
    if not patient.consent_asked:
        return False, "consent_not_asked"
    if patient.consent_declined:
        return False, "consent_declined"
    return True, ""
