"""CGM alarm generation.

Alarms are generated from the *displayed* CGM value, not from the latent truth,
so noise and artefacts produce false alarms exactly as they would on a real
dashboard. Two design points matter for the research question:

* An individualised hyperglycaemia threshold can be set for patients with
  chronic uncontrolled hyperglycaemia. This is the alarm-fatigue lever: without
  it those patients alarm constantly and the agent learns to ignore the board.
* Signal loss produces NO alarm. Missing data is only visible as a growing
  "steps since last reading" gap, which the agent has to notice unprompted.

ACADEMIC MODEL ONLY - simplified placeholders, not clinical guidance.
Stdlib only (ships to the browser build).
"""

from dataclasses import dataclass
from enum import Enum

from .patient import PatientState


class AlarmKind(str, Enum):
    HYPO = "hypo"
    SEVERE_HYPO = "severe_hypo"
    HYPER = "hyper"
    RAPID_FALL = "rapid_fall"
    RAPID_RISE = "rapid_rise"


# Alarms that the model treats as clinically significant, i.e. the agent is
# expected to confirm them with a point-of-care capillary test before treating.
SIGNIFICANT_ALARMS = (AlarmKind.HYPO, AlarmKind.SEVERE_HYPO, AlarmKind.HYPER)

# Persistence is tracked by clinical *family*, not by exact alarm kind. A
# patient oscillating 2.9, 3.1, 2.9 is continuously hypoglycaemic, but the
# readings alternate between the severe and plain kinds - so a per-kind streak
# would never reach two and the alarm would never fire.
ALARM_FAMILIES = {
    AlarmKind.HYPO: "hypo",
    AlarmKind.SEVERE_HYPO: "hypo",
    AlarmKind.HYPER: "hyper",
    AlarmKind.RAPID_FALL: "rapid_fall",
    AlarmKind.RAPID_RISE: "rapid_rise",
}


def alarm_family(kind: AlarmKind) -> str:
    return ALARM_FAMILIES[kind]

# Alarms that demand immediate attention; doing administrative work while one
# of these is live is penalised as unsafe prioritisation.
URGENT_ALARMS = (AlarmKind.HYPO, AlarmKind.SEVERE_HYPO, AlarmKind.RAPID_FALL)


@dataclass
class Alarm:
    """A live alarm on the telemetry dashboard."""

    bed: int
    kind: AlarmKind
    raised_step: int
    cgm_value: float
    acknowledged_step: int | None = None
    poc_confirmed_step: int | None = None
    poc_value: float | None = None
    resolved_step: int | None = None
    # True when the latent glucose did not actually breach the threshold, i.e.
    # the alarm was driven by sensor noise or an artefact.
    false_alarm: bool = False
    counted_delay: bool = False

    @property
    def is_urgent(self) -> bool:
        return self.kind in URGENT_ALARMS

    @property
    def needs_poc(self) -> bool:
        return self.kind in SIGNIFICANT_ALARMS

    def age(self, step: int) -> int:
        return step - self.raised_step

    def response_time(self) -> int | None:
        if self.acknowledged_step is None:
            return None
        return self.acknowledged_step - self.raised_step


def rate_of_change(patient: PatientState, window: int) -> float | None:
    """Change in *smoothed* displayed CGM over the last ``window`` steps.

    Real CGM devices smooth the signal before computing a trend, and for good
    reason: differencing two raw samples doubles the noise, so an unsmoothed
    trend alarm fires constantly on sensor noise alone. Averaging pairs of
    adjacent samples before differencing is the cheapest version of that.
    """
    history = patient.cgm_history
    if len(history) <= window + 1:
        return None
    recent = (history[-1] + history[-2]) / 2.0
    earlier = (history[-1 - window] + history[-2 - window]) / 2.0
    return recent - earlier


def evaluate_alarms(
    patient: PatientState,
    cgm_value: float | None,
    step: int,
    cfg,
) -> list[AlarmKind]:
    """Return the alarm kinds this patient's latest CGM value should raise.

    Returns an empty list when there is no signal - silent failure is the point.
    """
    if cgm_value is None or not patient.is_enrolled:
        return []

    ac = cfg.alarms
    kinds: list[AlarmKind] = []

    if cgm_value < ac.severe_hypo_threshold:
        kinds.append(AlarmKind.SEVERE_HYPO)
    elif cgm_value < ac.hypo_threshold:
        kinds.append(AlarmKind.HYPO)

    hyper_threshold = patient.hyper_threshold(ac.hyper_threshold_default)
    if cgm_value > hyper_threshold:
        kinds.append(AlarmKind.HYPER)

    roc = rate_of_change(patient, ac.roc_window_steps)
    if roc is not None:
        if roc <= ac.rapid_fall_threshold:
            kinds.append(AlarmKind.RAPID_FALL)
        elif roc >= ac.rapid_rise_threshold:
            kinds.append(AlarmKind.RAPID_RISE)

    return kinds


def is_false_alarm(patient: PatientState, kind: AlarmKind, cfg) -> bool:
    """Whether the latent state clearly contradicts the alarm that was raised.

    A margin is applied so that a patient sitting right on a threshold is not
    counted as a false alarm every time noise nudges the displayed value across
    it - that would be an artefact of the definition, not alarm burden.
    """
    ac = cfg.alarms
    margin = ac.false_alarm_margin
    true_g = patient.true_glucose
    if kind is AlarmKind.SEVERE_HYPO:
        return true_g >= ac.severe_hypo_threshold + margin
    if kind is AlarmKind.HYPO:
        return true_g >= ac.hypo_threshold + margin
    if kind is AlarmKind.HYPER:
        return true_g <= patient.hyper_threshold(ac.hyper_threshold_default) - margin
    # Rate-of-change alarms are judged against the latent trajectory.
    history = patient.glucose_history
    window = ac.roc_window_steps
    if len(history) <= window:
        return True
    true_roc = history[-1] - history[-1 - window]
    # A trend alarm on a patient who is genuinely out of range is not a
    # nuisance alarm even if the trend itself was overstated.
    if true_g < ac.hypo_threshold or true_g > patient.hyper_threshold(
        ac.hyper_threshold_default
    ):
        return False
    if kind is AlarmKind.RAPID_FALL:
        return true_roc > ac.rapid_fall_threshold / 2
    return true_roc < ac.rapid_rise_threshold / 2
