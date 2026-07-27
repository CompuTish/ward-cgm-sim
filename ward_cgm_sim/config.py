"""Central configuration for the ward CGM telemetry simulator.

ACADEMIC MODEL ONLY. Every clinical number in this file is a simplified,
guideline-inspired placeholder chosen to make the simulation behave plausibly.
None of it is prescribing guidance, bedside instruction, or clinical decision
support. Thresholds, treatment effects and timings are configurable precisely
so that a researcher can substitute values appropriate to their own protocol.

Stdlib only: this module is shipped into the browser (WebAssembly) build, so it
must never import numpy, gymnasium or any other native dependency.
"""

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Time base
# --------------------------------------------------------------------------
MINUTES_PER_STEP = 5
SHIFT_HOURS = 12
STEPS_PER_EPISODE = SHIFT_HOURS * 60 // MINUTES_PER_STEP  # 144


@dataclass
class GlucoseConfig:
    """True (latent) glucose dynamics and the sensing chain on top of it.

    Units are mmol/L throughout, matching UK inpatient practice.
    """

    # Ornstein-Uhlenbeck style mean reversion for the latent glucose signal.
    reversion_rate: float = 0.06  # per 5-min step, toward the patient's target
    process_noise: float = 0.22  # mmol/L per step
    min_glucose: float = 1.2
    max_glucose: float = 33.0

    # Insulin and meal perturbations. Both act over a ramp rather than in a
    # single step: insulin does not drop glucose by 2 mmol/L in five minutes,
    # and modelling it that way would make the trajectory impossible for any
    # sensor to track and rate-of-change alarms meaningless.
    meal_prob: float = 0.030
    meal_effect: float = 2.8
    meal_onset_steps: int = 6  # ~30 minutes
    insulin_prob: float = 0.035
    insulin_effect: float = -2.6
    insulin_onset_steps: int = 12  # ~1 hour

    # Deterioration episodes: an unlucky patient drifts toward hypo or hyper.
    hypo_episode_prob: float = 0.0012  # scaled by the patient's hypo risk
    hyper_episode_prob: float = 0.004  # scaled by the patient's hyper risk
    episode_drift: float = 0.45  # mmol/L per step while an episode is active
    episode_steps: tuple[int, int] = (6, 16)

    # CGM sensing chain. CGM is deliberately NOT ground truth.
    cgm_lag_steps: int = 2  # interstitial lag (~10 min)
    cgm_noise_sd: float = 0.45  # baseline Gaussian noise (MARD-inspired)
    cgm_bias_sd: float = 0.45  # per-sensor fixed bias, drawn at insertion
    cgm_degraded_noise_multiplier: float = 3.0
    cgm_spike_prob: float = 0.005  # transient artefact -> possible false alarm
    cgm_spike_magnitude: float = 4.0

    # Sensor reliability. Signal loss is SILENT: it produces no alarm, only a
    # gap in the data that the agent has to notice for itself.
    sensor_degrade_prob: float = 0.004  # per enrolled patient per step
    sensor_signal_loss_prob: float = 0.005
    signal_loss_steps: tuple[int, int] = (4, 30)
    troubleshoot_success_prob: float = 0.75

    # Point-of-care capillary meter: much more accurate, still not perfect.
    poc_noise_sd: float = 0.28

    # Treatment effects (simplified placeholders, guideline-inspired only).
    hypo_treatment_effect: float = 3.6  # total rise delivered over the ramp
    hypo_treatment_steps: int = 3
    hyper_treatment_effect: float = -3.2
    hyper_treatment_steps: int = 4
    escalation_effect_multiplier: float = 1.5  # specialist input works better


@dataclass
class AlarmConfig:
    """CGM alarm thresholds and rate-of-change rules."""

    hypo_threshold: float = 3.9  # mmol/L, standard inpatient hypoglycaemia cut-off
    severe_hypo_threshold: float = 3.0  # stronger safety penalty below this
    # Minimum time below threshold for something to count as a hypoglycaemic
    # *event* rather than a transient dip, following the consensus 15-minute
    # definition used for CGM-derived metrics. Three 5-minute steps.
    hypo_event_min_steps: int = 3
    hyper_threshold_default: float = 14.0
    # Patients with chronic uncontrolled hyperglycaemia can be given a higher
    # personal threshold to reduce nuisance alarms (alarm-fatigue lever).
    hyper_threshold_individualised: float = 18.0
    # Share of chronically hyperglycaemic patients given the raised threshold.
    # The rest are the residual nuisance-alarm burden the agent has to live
    # with, which is what makes alarm fatigue visible in the results.
    individualised_threshold_fraction: float = 0.7

    # Rate-of-change alarms, evaluated over a 15-minute (3 step) window.
    roc_window_steps: int = 3
    rapid_fall_threshold: float = -1.8  # mmol/L over the window
    rapid_rise_threshold: float = 2.6

    # Consecutive out-of-range readings required before an alarm is raised.
    # Real CGM alarm systems use persistence logic like this to suppress
    # single-sample artefacts; it is one of the main levers available for
    # trading alarm burden against detection latency, so it is configurable.
    persistence_readings: int = 2

    # An alarm counts as a false alarm only when the latent glucose is clearly
    # on the other side of the threshold. Without this margin, a patient
    # genuinely hovering at 13.9 mmol/L who alarms at 14.1 would be scored as a
    # false alarm, which is a definitional artefact rather than a real one.
    false_alarm_margin: float = 0.8

    # An alarm must be re-armed before it can fire again for the same patient.
    realarm_cooldown_steps: int = 6
    # Steps after which an unanswered alarm counts as a delayed response.
    response_deadline_steps: int = 3
    # Steps of missing CGM data after which ignoring it is penalised.
    signal_loss_grace_steps: int = 6


@dataclass
class PatientConfig:
    """Population mix and eligibility-relevant characteristics."""

    diabetes_prevalence: float = 0.40  # inflated vs a real ward, for signal density
    # Telemetry is modelled as ongoing routine care, not something started from
    # scratch each shift: a share of the eligible patients already on the ward
    # at handover are enrolled with sensors running. The agent inherits that
    # cohort and has to manage it, which is the workflow question of interest.
    initial_enrolled_fraction: float = 0.6
    type_weights: dict[str, float] = field(
        default_factory=lambda: {"type1": 0.22, "type2": 0.65, "type3c": 0.08, "other": 0.05}
    )
    surgical_fraction: float = 0.45  # mixed medical/surgical ward

    # Insulin regimen (only patients on >=2 injections/day are eligible).
    prob_two_or_more_injections_if_diabetic: float = 0.55
    # Probability that an enrolled patient's regimen is reduced mid-shift,
    # which makes them ineligible and requires de-enrolment.
    regimen_reduction_prob: float = 0.0035

    # Consent and capacity.
    prob_has_capacity: float = 0.88
    prob_consents_if_asked: float = 0.82

    # Length of stay in hours; >=48h is required for enrolment. Calibrated so
    # that a 32-bed ward turns over roughly five or six patients per 12-hour
    # shift, which is the order of magnitude a real acute ward runs at.
    los_hours_range: tuple[float, float] = (12.0, 168.0)
    prob_los_at_least_48h: float = 0.55

    # Exclusions.
    prob_pregnant_or_breastfeeding: float = 0.03
    prob_end_of_life: float = 0.04
    prob_becomes_end_of_life: float = 0.0015  # per step, transitions mid-shift

    # Risk profile.
    hypo_risk_range: tuple[float, float] = (0.1, 1.0)
    hyper_risk_range: tuple[float, float] = (0.1, 1.0)
    # Usual glucose for a patient with diabetes, interpolated by hyper_risk:
    # well-controlled patients sit near the bottom, chronically uncontrolled
    # patients near the top and therefore close to the alarm threshold.
    target_glucose_range: tuple[float, float] = (6.0, 15.0)


@dataclass
class WardConfig:
    """Bed flow, admissions and discharge pipeline."""

    n_beds: int = 32
    initial_occupancy: float = 0.88
    initial_queue: tuple[int, int] = (2, 6)

    # Time-varying arrival intensity (expected arrivals per step). Tuned so a
    # competent policy can keep the queue safe but a distracted one cannot,
    # and so that admissions roughly balance discharges over a shift.
    arrival_rate_base: float = 0.035
    arrival_rate_peak: float = 0.07
    peak_start_step: int = 36  # arrivals build through the middle of the shift
    peak_end_step: int = 108

    # Discharge pipeline: ready -> reviewed -> supported -> discharged.
    # Background staff move patients along on their own, slowly; the agent's
    # SUPPORT_DISCHARGE action is what makes it fast. A ward that only
    # discharged when the shift coordinator personally intervened would grind
    # to a halt, which is not how a real ward behaves.
    # Discharge readiness is driven by how far through their expected stay a
    # patient is, not by a flat hazard. This keeps the model self-consistent:
    # expected length of stay is an eligibility criterion, so a patient with a
    # documented 48-hour-plus stay must not evaporate an hour later.
    discharge_ready_prob: float = 0.06  # per step once at/after expected LOS
    discharge_ready_early_fraction: float = 0.85  # readiness can start this far in
    discharge_review_decay: int = 24  # steps a review stays "fresh"
    discharge_steps_after_support: int = 3
    background_review_prob: float = 0.05  # ready -> reviewed without the agent
    background_support_prob: float = 0.035  # reviewed -> supported without the agent

    # Overcrowding.
    safe_queue_length: int = 4
    unsafe_queue_length: int = 14  # episode terminates: unsafe overcrowding
    overcrowding_penalty_start: int = 6

    # Transfers off the ward (imaging, theatre) make a patient temporarily
    # unavailable and visibly absent from their bed.
    transfer_prob: float = 0.004
    transfer_steps: tuple[int, int] = (4, 18)


@dataclass
class UsualCareConfig:
    """The baseline ward monitoring that exists with or without telemetry.

    Patients who are not on CGM are not unmonitored: routine capillary glucose
    rounds and symptom recognition still catch deteriorating patients, just
    more slowly and less reliably. Modelling this matters enormously for the
    research question - without it, "no telemetry" would mean "no monitoring at
    all", and CGM would look better than it could possibly be in reality.

    The comparison the simulator is actually built to make is therefore
    *telemetry versus routine monitoring*, not telemetry versus nothing.
    """

    # Routine monitoring applies to EVERY patient, including those on
    # telemetry: CGM is additive to standard ward care, not a replacement for
    # it. Detection is therefore whichever route notices first.
    #
    # Per-step probability that routine care notices a patient below the
    # hypoglycaemia threshold. Capillary rounds run roughly 4-6 hourly, i.e.
    # every 48-72 five-minute steps, so an unremarkable low is usually only
    # found at the next scheduled check.
    routine_detection_prob: float = 0.02
    # Severe hypoglycaemia is far more likely to be noticed between checks
    # because the patient becomes symptomatic.
    severe_detection_multiplier: float = 3.0
    # Detection is worse when the ward is short-staffed.
    understaffed_multiplier: float = 0.5
    # Once noticed, background staff treat it. This is usual care, so it does
    # not earn the agent any reward - it is the baseline being compared against.
    treat_on_detection: bool = True
    # Steps of continuous untreated severe hypoglycaemia before the model
    # records a serious adverse event (18 steps = 90 minutes). Placeholder.
    sae_untreated_steps: int = 18


@dataclass
class StaffConfig:
    """Background staff availability (hidden until the agent asks)."""

    roles: tuple[str, ...] = ("hca", "nurse", "doctor", "surgeon", "diabetes")
    # Hidden two-state Markov chain per role.
    availability_start: dict[str, float] = field(
        default_factory=lambda: {
            "hca": 0.75,
            "nurse": 0.70,
            "doctor": 0.55,
            "surgeon": 0.40,
            "diabetes": 0.45,
        }
    )
    become_available_prob: float = 0.12
    become_busy_prob: float = 0.10
    # Workload accumulates when the agent leans on colleagues.
    workload_per_request: float = 1.0
    workload_decay: float = 0.06
    workload_penalty_threshold: float = 12.0


@dataclass
class RewardConfig:
    """Reward weights. Safety first, then workflow and bed pressure.

    Positive weights are rewards, negative weights are penalties. All values
    are exposed here so that sensitivity analysis is a config change rather
    than a code change.
    """

    # --- Large positive: clinical safety -----------------------------------
    hypo_treated_promptly: float = 10.0
    hypo_prevented: float = 6.0  # acted before the patient fell below range
    correct_escalation: float = 8.0
    enrolled_patient_safe_per_step: float = 0.02
    safe_occupancy_per_step: float = 0.05
    shift_completed_without_sae: float = 15.0

    # --- Moderate positive: enrolment quality and ward workflow ------------
    correct_enrolment: float = 5.0
    correct_ineligible_identification: float = 2.0
    correct_deenrolment: float = 4.0
    fast_alarm_response: float = 2.0
    discharge_supported: float = 3.0
    queue_reduced: float = 0.5
    alarm_fatigue_bonus_max: float = 3.0  # end-of-shift, scales with nuisance rate

    # --- Negative: safety failures -----------------------------------------
    serious_adverse_event: float = -50.0  # terminal
    missed_severe_hypo: float = -20.0
    time_below_range_per_step: float = -0.5
    delayed_alarm_response_per_step: float = -0.2
    wrong_patient_treatment: float = -4.0
    treatment_without_poc_confirmation: float = -2.0
    unnecessary_treatment_poc_normal: float = -3.0
    ignored_signal_loss_per_step: float = -0.3
    unsafe_prioritisation_per_step: float = -1.0

    # --- Negative: enrolment errors ----------------------------------------
    enrolled_ineligible: float = -6.0
    missed_eligible_patient: float = -3.0
    failure_to_deenrol_per_step: float = -0.2
    unnecessary_deenrolment: float = -4.0

    # --- Negative: workflow and bed pressure -------------------------------
    staff_overload: float = -0.2
    discharge_delay_per_step: float = -0.1
    queue_per_patient_per_step: float = -0.05
    overcrowding_per_step: float = -1.0
    unsafe_overcrowding: float = -30.0  # terminal
    invalid_action: float = -0.05
    wrong_role_request: float = -0.3


@dataclass
class SimConfig:
    """Top-level simulation configuration."""

    seed: int | None = None
    steps_per_episode: int = STEPS_PER_EPISODE
    minutes_per_step: int = MINUTES_PER_STEP

    # The counterfactual switch that drives the research question: with
    # telemetry disabled there is no dashboard and no alarms, so the agent can
    # only find deteriorating patients by physically checking them.
    telemetry_enabled: bool = True

    # Terminating the episode on a serious adverse event makes safety failures
    # unambiguous; set False to let an episode run its full length for analysis.
    terminate_on_sae: bool = True
    terminate_on_unsafe_overcrowding: bool = True

    glucose: GlucoseConfig = field(default_factory=GlucoseConfig)
    alarms: AlarmConfig = field(default_factory=AlarmConfig)
    patients: PatientConfig = field(default_factory=PatientConfig)
    ward: WardConfig = field(default_factory=WardConfig)
    staff: StaffConfig = field(default_factory=StaffConfig)
    usual_care: UsualCareConfig = field(default_factory=UsualCareConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def step_minutes(self, steps: int) -> int:
        return steps * self.minutes_per_step
