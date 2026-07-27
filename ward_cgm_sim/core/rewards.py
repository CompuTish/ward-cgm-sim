"""Reward accounting.

Every reward component is named and accumulated separately so an experiment can
report *why* a policy scored what it did - which matters far more for the
research question than the scalar return. ``RewardTracker.components`` is
surfaced in the Gymnasium ``info`` dict at the end of each episode.

Weights live in ``config.RewardConfig``; nothing here hard-codes a number.

Stdlib only (ships to the browser build).
"""

from collections import defaultdict


class RewardTracker:
    """Accumulates per-step reward and keeps a named breakdown."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = cfg.rewards
        self.components: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.step_reward: float = 0.0
        self.total: float = 0.0

    # ------------------------------------------------------------------
    def begin_step(self) -> None:
        self.step_reward = 0.0

    def add(self, name: str, weight: float, multiplier: float = 1.0) -> float:
        """Record ``weight * multiplier`` against the named component."""
        value = weight * multiplier
        self.components[name] += value
        self.counts[name] += 1
        self.step_reward += value
        self.total += value
        return value

    def end_step(self) -> float:
        return self.step_reward

    # ------------------------------------------------------------------
    # Named helpers. Each maps one spec bullet to one weight, so the mapping
    # from the written reward specification to code is auditable line by line.
    # ------------------------------------------------------------------

    # --- large positive: safety ---------------------------------------
    def hypo_treated_promptly(self, multiplier: float = 1.0) -> None:
        self.add("hypo_treated_promptly", self.weights.hypo_treated_promptly, multiplier)

    def hypo_prevented(self) -> None:
        self.add("hypo_prevented", self.weights.hypo_prevented)

    def correct_escalation(self) -> None:
        self.add("correct_escalation", self.weights.correct_escalation)

    def enrolled_patients_safe(self, n_safe: int) -> None:
        if n_safe:
            self.add("enrolled_patient_safe", self.weights.enrolled_patient_safe_per_step, n_safe)

    def safe_occupancy(self) -> None:
        self.add("safe_occupancy", self.weights.safe_occupancy_per_step)

    def shift_completed_without_sae(self) -> None:
        self.add("shift_completed_without_sae", self.weights.shift_completed_without_sae)

    # --- moderate positive: enrolment quality and workflow ------------
    def correct_enrolment(self) -> None:
        self.add("correct_enrolment", self.weights.correct_enrolment)

    def correct_ineligible_identification(self) -> None:
        self.add(
            "correct_ineligible_identification",
            self.weights.correct_ineligible_identification,
        )

    def correct_deenrolment(self) -> None:
        self.add("correct_deenrolment", self.weights.correct_deenrolment)

    def fast_alarm_response(self, multiplier: float = 1.0) -> None:
        self.add("fast_alarm_response", self.weights.fast_alarm_response, multiplier)

    def discharge_supported(self) -> None:
        self.add("discharge_supported", self.weights.discharge_supported)

    def queue_reduced(self, n: int) -> None:
        if n:
            self.add("queue_reduced", self.weights.queue_reduced, n)

    def alarm_fatigue_bonus(self, fraction_avoided: float) -> None:
        """End-of-shift bonus scaling with how few nuisance alarms were left live.

        Rewards individualised hyperglycaemia thresholds and prompt sensor
        troubleshooting, both of which cut the nuisance-alarm burden.
        """
        self.add(
            "alarm_fatigue_bonus",
            self.weights.alarm_fatigue_bonus_max,
            max(0.0, min(1.0, fraction_avoided)),
        )

    # --- negative: safety failures ------------------------------------
    def serious_adverse_event(self) -> None:
        self.add("serious_adverse_event", self.weights.serious_adverse_event)

    def missed_severe_hypo(self) -> None:
        self.add("missed_severe_hypo", self.weights.missed_severe_hypo)

    def time_below_range(self, n_patients: int) -> None:
        if n_patients:
            self.add("time_below_range", self.weights.time_below_range_per_step, n_patients)

    def delayed_alarm_response(self, n_alarms: int) -> None:
        if n_alarms:
            self.add(
                "delayed_alarm_response",
                self.weights.delayed_alarm_response_per_step,
                n_alarms,
            )

    def wrong_patient_treatment(self) -> None:
        self.add("wrong_patient_treatment", self.weights.wrong_patient_treatment)

    def treatment_without_poc(self) -> None:
        self.add(
            "treatment_without_poc_confirmation",
            self.weights.treatment_without_poc_confirmation,
        )

    def unnecessary_treatment(self) -> None:
        self.add(
            "unnecessary_treatment_poc_normal",
            self.weights.unnecessary_treatment_poc_normal,
        )

    def ignored_signal_loss(self, n_patients: int) -> None:
        if n_patients:
            self.add(
                "ignored_signal_loss",
                self.weights.ignored_signal_loss_per_step,
                n_patients,
            )

    def unsafe_prioritisation(self) -> None:
        self.add("unsafe_prioritisation", self.weights.unsafe_prioritisation_per_step)

    # --- negative: enrolment errors -----------------------------------
    def enrolled_ineligible(self) -> None:
        self.add("enrolled_ineligible", self.weights.enrolled_ineligible)

    def missed_eligible_patient(self) -> None:
        self.add("missed_eligible_patient", self.weights.missed_eligible_patient)

    def failure_to_deenrol(self, n_patients: int) -> None:
        if n_patients:
            self.add(
                "failure_to_deenrol",
                self.weights.failure_to_deenrol_per_step,
                n_patients,
            )

    def unnecessary_deenrolment(self) -> None:
        self.add("unnecessary_deenrolment", self.weights.unnecessary_deenrolment)

    # --- negative: workflow and bed pressure --------------------------
    def staff_overload(self) -> None:
        self.add("staff_overload", self.weights.staff_overload)

    def discharge_delay(self, n_patients: int) -> None:
        if n_patients:
            self.add("discharge_delay", self.weights.discharge_delay_per_step, n_patients)

    def queue_pressure(self, queue_length: int) -> None:
        if queue_length:
            self.add(
                "queue_pressure",
                self.weights.queue_per_patient_per_step,
                queue_length,
            )

    def overcrowding(self) -> None:
        self.add("overcrowding", self.weights.overcrowding_per_step)

    def unsafe_overcrowding(self) -> None:
        self.add("unsafe_overcrowding", self.weights.unsafe_overcrowding)

    def invalid_action(self) -> None:
        self.add("invalid_action", self.weights.invalid_action)

    def wrong_role_request(self) -> None:
        self.add("wrong_role_request", self.weights.wrong_role_request)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, float]:
        return dict(self.components)
