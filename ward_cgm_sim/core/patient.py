"""Patient state model.

The split between what is *true* about a patient and what the agent *knows*
about them is the heart of the POMDP. Everything in ``PatientState`` is hidden
unless the agent has done something to learn it; ``PatientKnowledge`` records
what has been learned and when, so that stale information can decay.

ACADEMIC MODEL ONLY - simplified placeholders, not clinical guidance.
Stdlib only (ships to the browser build).
"""

from dataclasses import dataclass, field
from enum import Enum
import random


class DiabetesType(str, Enum):
    NONE = "none"
    TYPE1 = "type1"
    TYPE2 = "type2"
    TYPE3C = "type3c"
    OTHER = "other"


class Specialty(str, Enum):
    """Mixed medical/surgical ward: specialty drives escalation routing."""

    MEDICAL = "medical"
    SURGICAL = "surgical"


class Location(str, Enum):
    BED = "bed"
    WALKING = "walking"  # visibly moving on the map (admission/transfer/discharge)
    OFF_WARD = "off_ward"  # imaging, theatre - not on the map
    DISCHARGED = "discharged"


class EnrolmentStatus(str, Enum):
    NOT_ENROLLED = "not_enrolled"
    ENROLLED = "enrolled"
    DECLINED = "declined"
    DEENROLLED = "deenrolled"


class DischargeStage(str, Enum):
    NOT_READY = "not_ready"
    READY = "ready"  # clinically ready, nobody has reviewed it yet
    REVIEWED = "reviewed"  # agent or doctor has confirmed readiness
    SUPPORTED = "supported"  # discharge paperwork/logistics under way
    DISCHARGED = "discharged"


UNKNOWN = -1  # sentinel used in observations for "the agent has not looked"


@dataclass
class PatientKnowledge:
    """What the agent has learned about a patient, and when.

    Anything left at ``None`` has never been observed and is encoded as
    ``UNKNOWN`` in the observation vector.
    """

    eligibility_reviewed_step: int | None = None
    known_has_diabetes: bool | None = None
    known_two_or_more_injections: bool | None = None
    known_expected_los_at_least_48h: bool | None = None
    known_has_capacity: bool | None = None
    consent_asked_step: int | None = None
    known_consented: bool | None = None
    known_excluded: bool | None = None  # pregnancy/breastfeeding/end-of-life seen in notes
    last_checked_step: int | None = None
    last_poc_step: int | None = None
    last_poc_value: float | None = None
    discharge_reviewed_step: int | None = None
    known_discharge_ready: bool | None = None

    def knows_full_eligibility(self) -> bool:
        return (
            self.known_has_diabetes is not None
            and self.known_two_or_more_injections is not None
            and self.known_expected_los_at_least_48h is not None
            and self.known_has_capacity is not None
            and self.known_excluded is not None
        )


@dataclass
class PatientState:
    """Ground truth for one patient. Hidden from the agent by default.

    Each patient carries their own random streams (common random numbers). This
    is what makes the telemetry counterfactual sound: an intervention on one
    patient changes how many draws *that* patient consumes, but cannot shift
    anybody else's trajectory. With a single shared stream, supporting one
    discharge would silently rewrite the whole rest of the ward.
    """

    patient_id: int
    bed: int
    specialty: Specialty
    diabetes_type: DiabetesType
    insulin_injections_per_day: int
    has_capacity: bool
    will_consent: bool
    expected_los_hours: float
    pregnant_or_breastfeeding: bool
    end_of_life: bool
    hypo_risk: float
    hyper_risk: float
    target_glucose: float
    true_glucose: float
    individualised_hyper_threshold: float | None = None

    # --- dynamic state ----------------------------------------------------
    location: Location = Location.BED
    # True for patients selected for telemetry at handover. Set identically in
    # both arms so the counterfactual can report outcomes for the same cohort.
    telemetry_cohort: bool = False
    enrolment: EnrolmentStatus = EnrolmentStatus.NOT_ENROLLED
    enrolled_step: int | None = None
    discharge_stage: DischargeStage = DischargeStage.NOT_READY
    discharge_ready_step: int | None = None
    admitted_step: int = 0
    steps_on_ward: int = 0

    # Glucose machinery
    glucose_history: list[float] = field(default_factory=list)
    # Queued gradual effects as [per_step_delta, steps_remaining] pairs, used
    # for meals and insulin so that neither lands as an instantaneous jump.
    pending_effects: list[list[float]] = field(default_factory=list)
    active_episode: str | None = None  # "hypo" | "hyper" | None
    episode_steps_left: int = 0
    treatment_effect_remaining: float = 0.0
    treatment_steps_left: int = 0
    treatment_kind: str | None = None
    last_treatment_step: int | None = None
    treated_without_poc: bool = False

    # Sensor machinery (only meaningful while enrolled). ``sensor_bias`` stays
    # None until a sensor is fitted, so that the draw comes from the sensor
    # random stream rather than the physiology one.
    sensor_bias: float | None = None
    sensor_degraded: bool = False
    signal_lost: bool = False
    signal_loss_steps_left: int = 0
    steps_since_valid_cgm: int = 0
    last_cgm_value: float | None = None
    cgm_history: list[float] = field(default_factory=list)
    # Consecutive out-of-range readings per alarm kind, for persistence logic.
    alarm_streak: dict[str, int] = field(default_factory=dict)

    # Transfers and movement
    transfer_steps_left: int = 0
    walk_steps_left: int = 0
    walk_total_steps: int = 1
    walk_purpose: str | None = None
    # Separate from walk_steps_left on purpose: discharge paperwork runs down
    # while the patient is still in bed, and only then do they walk off.
    discharge_prep_steps_left: int = 0

    # Bookkeeping for rewards
    consent_asked: bool = False
    consent_declined: bool = False
    became_ineligible_step: int | None = None
    counted_missed_eligible: bool = False
    severe_hypo_untreated_steps: int = 0
    severe_episode_active: bool = False
    sae_recorded: bool = False
    # Detection-latency tracking: when this hypoglycaemic episode began, and
    # whether anybody has found it yet. The primary outcome measure.
    hypo_episode_started_step: int | None = None
    hypo_episode_detected: bool = False
    hypo_episode_counted: bool = False
    hypo_episode_detected_step: int | None = None
    hypo_episode_detected_route: str | None = None

    knowledge: PatientKnowledge = field(default_factory=PatientKnowledge)

    # Per-patient streams, seeded deterministically from the episode seed and
    # this patient's id. ``rng`` drives their physiology, transfers and
    # discharge progression; ``rng_sensor`` drives their CGM chain and the
    # routine-monitoring checks on them.
    # ``rng_care`` is kept separate from ``rng_sensor`` so that a patient's
    # routine-monitoring checks stay aligned across the telemetry and
    # counterfactual arms even though only one arm consumes sensor draws.
    # ``rng_action`` carries the consequences of the agent acting on THIS
    # patient, so that treating one patient cannot shift another's outcomes.
    rng: random.Random = field(default_factory=random.Random, repr=False)
    rng_sensor: random.Random = field(default_factory=random.Random, repr=False)
    rng_care: random.Random = field(default_factory=random.Random, repr=False)
    rng_action: random.Random = field(default_factory=random.Random, repr=False)

    # ------------------------------------------------------------------
    # Ground-truth predicates (hidden - used by the environment, never
    # exposed directly in the observation).
    # ------------------------------------------------------------------
    @property
    def has_diabetes(self) -> bool:
        return self.diabetes_type is not DiabetesType.NONE

    @property
    def expected_los_at_least_48h(self) -> bool:
        return self.expected_los_hours >= 48.0

    @property
    def two_or_more_injections(self) -> bool:
        return self.insulin_injections_per_day >= 2

    @property
    def is_enrolled(self) -> bool:
        return self.enrolment is EnrolmentStatus.ENROLLED

    @property
    def on_ward(self) -> bool:
        return self.location in (Location.BED, Location.WALKING)

    @property
    def visible_at_bed(self) -> bool:
        return self.location is Location.BED

    def hyper_threshold(self, default: float) -> float:
        return self.individualised_hyper_threshold or default


def sample_patient(
    rng: random.Random,
    patient_id: int,
    bed: int,
    cfg,
    step: int = 0,
    stream_seed: int = 0,
) -> PatientState:
    """Draw a new patient from the configured population mix.

    ``rng`` is the ward-level stream used to sample the patient's fixed
    characteristics. ``stream_seed`` seeds the patient's own streams, which
    from then on drive everything that happens *to* them.
    """
    pc = cfg.patients
    gc = cfg.glucose
    ac = cfg.alarms

    has_diabetes = rng.random() < pc.diabetes_prevalence
    if has_diabetes:
        types = list(pc.type_weights.keys())
        weights = [pc.type_weights[t] for t in types]
        diabetes_type = DiabetesType(rng.choices(types, weights=weights, k=1)[0])
        injections = 2 + rng.randint(0, 2) if rng.random() < pc.prob_two_or_more_injections_if_diabetic else rng.randint(0, 1)
    else:
        diabetes_type = DiabetesType.NONE
        injections = 0

    if rng.random() < pc.prob_los_at_least_48h:
        expected_los = rng.uniform(48.0, pc.los_hours_range[1])
    else:
        expected_los = rng.uniform(pc.los_hours_range[0], 47.0)

    specialty = Specialty.SURGICAL if rng.random() < pc.surgical_fraction else Specialty.MEDICAL
    # Surgical patients tend to have shorter planned stays.
    if specialty is Specialty.SURGICAL and rng.random() < 0.25:
        expected_los = min(expected_los, rng.uniform(12.0, 47.0))

    hypo_risk = rng.uniform(*pc.hypo_risk_range) if has_diabetes else rng.uniform(0.0, 0.2)
    hyper_risk = rng.uniform(*pc.hyper_risk_range) if has_diabetes else rng.uniform(0.0, 0.2)
    if has_diabetes:
        # A patient's usual glucose tracks how poorly controlled they are, so
        # the chronically hyperglycaemic sit near the alarm threshold. Those
        # are exactly the patients an individualised threshold exists for.
        low, high = pc.target_glucose_range
        target = low + hyper_risk * (high - low)
    else:
        target = rng.uniform(4.5, 7.0)

    individualised = None
    if has_diabetes and hyper_risk > 0.65 and rng.random() < ac.individualised_threshold_fraction:
        individualised = ac.hyper_threshold_individualised

    patient = PatientState(
        patient_id=patient_id,
        bed=bed,
        specialty=specialty,
        diabetes_type=diabetes_type,
        insulin_injections_per_day=injections,
        has_capacity=rng.random() < pc.prob_has_capacity,
        will_consent=rng.random() < pc.prob_consents_if_asked,
        expected_los_hours=expected_los,
        pregnant_or_breastfeeding=rng.random() < pc.prob_pregnant_or_breastfeeding,
        end_of_life=rng.random() < pc.prob_end_of_life,
        hypo_risk=hypo_risk,
        hyper_risk=hyper_risk,
        target_glucose=target,
        true_glucose=max(gc.min_glucose, min(gc.max_glucose, rng.gauss(target, 1.4))),
        individualised_hyper_threshold=individualised,
        admitted_step=step,
    )
    # Domain-separated string seeds. Arithmetic seeding (XOR/multiply) collides:
    # with an episode seed of 0 it gave patient 7's physiology stream the same
    # state as patient 0's sensor stream, silently correlating patients and
    # domains that must be independent. Python hashes str seeds, so embedding
    # the domain name keeps every stream distinct.
    for name in ("rng", "rng_sensor", "rng_care", "rng_action"):
        setattr(patient, name, random.Random(f"ward-cgm-sim|{stream_seed}|{patient_id}|{name}"))
    patient.glucose_history.append(patient.true_glucose)
    return patient
