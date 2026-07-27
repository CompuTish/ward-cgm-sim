"""Observation construction: the observable slice of the ward state.

This module is where the "partially observable" in POMDP is enforced. Nothing
here may read a hidden field of ``PatientState`` directly - it may only read
what the agent has recorded in ``PatientKnowledge``, plus what is genuinely
visible from where the agent is standing.

Layout (flat list of floats, stable ordering):

  Ward block (8 values)
    0  fraction of the shift elapsed              [0, 1]
    1  bed occupancy fraction                     [0, 1]
    2  free beds / n_beds                         [0, 1]
    3  ED/admissions queue length / unsafe limit  [0, 1+]
    4  number of visible alarms / n_beds          [0, 1]
    5  coarse staff availability / 2              {0, 0.5, 1}
    6  agent x / map width                        [0, 1]
    7  agent y / map height                       [0, 1]

  Per bed (9 values x 32 beds)
    0  occupied                                   {0, 1}
    1  patient visible at bed                     {0, 1}  (0 = off ward/walking)
    2  enrolment status                           {0 not, 0.5 declined/de-enrolled, 1 enrolled}
    3  known eligibility summary                  {-1 unknown, 0 ineligible, 1 eligible}
    4  known consent status                       {-1 not asked, 0 declined, 1 consented}
    5  visible CGM value, normalised              {-1 unknown/no signal, else g/25}
    6  steps since a valid CGM reading            {-1 unknown, else min(steps, 20)/20}
    7  visible alarm severity for this bed        {0 none, 0.5 non-urgent, 1 urgent}
    8  known discharge status                     {-1 unreviewed, 0 not ready, 1 ready}

``UNKNOWN`` (-1) is used deliberately rather than a plausible default: an agent
must be able to tell "I have not looked" apart from "I looked and it was no".

Stdlib only (ships to the browser build).
"""

from .patient import DischargeStage, EnrolmentStatus, Location, PatientState

UNKNOWN = -1.0
WARD_FEATURES = 8
BED_FEATURES = 9
GLUCOSE_SCALE = 25.0
STALENESS_CAP = 20.0


def observation_size(n_beds: int) -> int:
    return WARD_FEATURES + BED_FEATURES * n_beds


def _enrolment_code(patient: PatientState) -> float:
    if patient.enrolment is EnrolmentStatus.ENROLLED:
        return 1.0
    if patient.enrolment in (EnrolmentStatus.DECLINED, EnrolmentStatus.DEENROLLED):
        return 0.5
    return 0.0


def _known_eligibility(patient: PatientState) -> float:
    """Summarises only what the agent has actually reviewed."""
    k = patient.knowledge
    if not k.knows_full_eligibility():
        return UNKNOWN
    eligible = (
        bool(k.known_has_diabetes)
        and bool(k.known_two_or_more_injections)
        and bool(k.known_expected_los_at_least_48h)
        and bool(k.known_has_capacity)
        and not bool(k.known_excluded)
    )
    return 1.0 if eligible else 0.0


def _known_consent(patient: PatientState) -> float:
    k = patient.knowledge
    if k.consent_asked_step is None:
        return UNKNOWN
    return 1.0 if k.known_consented else 0.0


def _known_discharge(patient: PatientState) -> float:
    k = patient.knowledge
    if k.discharge_reviewed_step is None:
        return UNKNOWN
    return 1.0 if k.known_discharge_ready else 0.0


def build_observation(engine) -> list[float]:
    """Assemble the flat observation vector from the engine's current state."""
    cfg = engine.cfg
    flow = engine.flow
    ward_map = engine.ward_map
    n_beds = flow.n_beds

    visible_alarms = engine.visible_alarms()

    obs: list[float] = [
        engine.step_index / max(1, cfg.steps_per_episode),
        flow.occupied_beds / n_beds,
        flow.free_beds / n_beds,
        min(1.5, flow.queue_length / max(1, cfg.ward.unsafe_queue_length)),
        len(visible_alarms) / n_beds,
        engine.staff.coarse_availability() / 2.0,
        engine.agent_x / ward_map.width,
        engine.agent_y / ward_map.height,
    ]

    alarm_by_bed: dict[int, float] = {}
    for alarm in visible_alarms:
        severity = 1.0 if alarm.is_urgent else 0.5
        alarm_by_bed[alarm.bed] = max(alarm_by_bed.get(alarm.bed, 0.0), severity)

    for bed in range(n_beds):
        patient = flow.patient_at_bed(bed)
        if patient is None:
            obs.extend([0.0, 0.0, 0.0, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, 0.0, UNKNOWN])
            continue

        visible_here = 1.0 if patient.location is Location.BED else 0.0

        # Glucose comes from the agent's last look at the telemetry board, not
        # from a live feed. An agent that has never checked has no glucose
        # information at all, and one that checked ten minutes ago is working
        # from ten-minute-old numbers.
        cgm_value = UNKNOWN
        staleness = UNKNOWN
        if cfg.telemetry_enabled and patient.is_enrolled:
            snapshot = engine.dashboard_snapshot.get(bed)
            # Identity check: a snapshot taken before this bed changed hands
            # belongs to the previous occupant and must not be shown.
            if snapshot is not None and snapshot[0] == patient.patient_id:
                _pid, seen_value, seen_staleness = snapshot
                age = engine.step_index - (engine.dashboard_seen_step or 0)
                if seen_value is not None:
                    cgm_value = min(1.0, seen_value / GLUCOSE_SCALE)
                staleness = min(STALENESS_CAP, seen_staleness + age) / STALENESS_CAP

        obs.extend(
            [
                1.0,
                visible_here,
                _enrolment_code(patient),
                _known_eligibility(patient),
                _known_consent(patient),
                cgm_value,
                staleness,
                alarm_by_bed.get(bed, 0.0),
                _known_discharge(patient),
            ]
        )

    return obs


def observation_labels(n_beds: int) -> list[str]:
    """Human-readable names, used by the renderer and for debugging."""
    labels = [
        "shift_fraction",
        "occupancy_fraction",
        "free_beds_fraction",
        "queue_fraction",
        "visible_alarm_fraction",
        "staff_availability",
        "agent_x",
        "agent_y",
    ]
    per_bed = [
        "occupied",
        "visible_at_bed",
        "enrolment",
        "known_eligibility",
        "known_consent",
        "cgm_value",
        "cgm_staleness",
        "alarm_severity",
        "known_discharge",
    ]
    for bed in range(n_beds):
        labels.extend(f"bed{bed:02d}_{name}" for name in per_bed)
    return labels


def discharge_stage_is_ready(patient: PatientState) -> bool:
    return patient.discharge_stage in (
        DischargeStage.READY,
        DischargeStage.REVIEWED,
        DischargeStage.SUPPORTED,
    )
