"""Latent glucose dynamics, CGM sensing and point-of-care capillary testing.

Three layers, deliberately distinct:

1. ``true_glucose``  - the latent physiological state. Nobody observes it.
2. CGM              - lagged, biased, noisy, occasionally artefactual, and
                      sometimes silently absent. This is what the telemetry
                      dashboard shows.
3. Point-of-care    - a capillary meter reading with much smaller error. In
                      this model PoC is the reference the agent is expected to
                      confirm clinically significant CGM alarms against, and it
                      is trusted over CGM when the two conflict.

ACADEMIC MODEL ONLY - simplified placeholders, not clinical guidance.
Stdlib only (ships to the browser build).
"""

import random

from .patient import PatientState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def step_true_glucose(patient: PatientState, rng: random.Random, cfg) -> None:
    """Advance one patient's latent glucose by one 5-minute step."""
    gc = cfg.glucose
    g = patient.true_glucose

    # Mean reversion toward the patient's individual target.
    g += gc.reversion_rate * (patient.target_glucose - g)

    # Meals and insulin, only meaningful for patients on insulin. Each event
    # queues a gradual effect rather than an instantaneous jump.
    if patient.has_diabetes:
        if rng.random() < gc.meal_prob:
            patient.pending_effects.append(
                [gc.meal_effect * rng.uniform(0.6, 1.4) / gc.meal_onset_steps, gc.meal_onset_steps]
            )
        if patient.insulin_injections_per_day >= 1 and rng.random() < gc.insulin_prob:
            patient.pending_effects.append(
                [
                    gc.insulin_effect * rng.uniform(0.6, 1.4) / gc.insulin_onset_steps,
                    gc.insulin_onset_steps,
                ]
            )

    # Apply and retire queued effects.
    if patient.pending_effects:
        still_active = []
        for effect in patient.pending_effects:
            g += effect[0]
            effect[1] -= 1
            if effect[1] > 0:
                still_active.append(effect)
        patient.pending_effects = still_active

    # Deterioration episodes.
    if patient.active_episode is None:
        if rng.random() < gc.hypo_episode_prob * patient.hypo_risk:
            patient.active_episode = "hypo"
            patient.episode_steps_left = rng.randint(*gc.episode_steps)
        elif rng.random() < gc.hyper_episode_prob * patient.hyper_risk:
            patient.active_episode = "hyper"
            patient.episode_steps_left = rng.randint(*gc.episode_steps)
    if patient.active_episode == "hypo":
        g -= gc.episode_drift
    elif patient.active_episode == "hyper":
        g += gc.episode_drift
    if patient.active_episode is not None:
        patient.episode_steps_left -= 1
        if patient.episode_steps_left <= 0:
            patient.active_episode = None

    # Treatment ramps in over several steps rather than instantly.
    if patient.treatment_steps_left > 0:
        per_step = patient.treatment_effect_remaining / patient.treatment_steps_left
        g += per_step
        patient.treatment_effect_remaining -= per_step
        patient.treatment_steps_left -= 1
        if patient.treatment_steps_left <= 0:
            patient.treatment_kind = None
            patient.treatment_effect_remaining = 0.0

    g += rng.gauss(0.0, gc.process_noise)
    patient.true_glucose = _clamp(g, gc.min_glucose, gc.max_glucose)
    patient.glucose_history.append(patient.true_glucose)
    if len(patient.glucose_history) > 64:
        del patient.glucose_history[:-64]


def apply_treatment(patient: PatientState, kind: str, cfg, escalated: bool = False) -> None:
    """Apply a simplified, guideline-inspired treatment pathway.

    This is a placeholder that moves glucose in the intended direction over a
    few steps. It is NOT a prescribing algorithm and carries no doses.
    """
    gc = cfg.glucose
    multiplier = gc.escalation_effect_multiplier if escalated else 1.0
    if kind == "hypo":
        patient.treatment_effect_remaining = gc.hypo_treatment_effect * multiplier
        patient.treatment_steps_left = gc.hypo_treatment_steps
    else:
        patient.treatment_effect_remaining = gc.hyper_treatment_effect * multiplier
        patient.treatment_steps_left = gc.hyper_treatment_steps
    patient.treatment_kind = kind


def step_sensor(patient: PatientState, rng: random.Random, cfg) -> float | None:
    """Advance the CGM sensor and return the displayed value, or None.

    Returning ``None`` models signal loss. Crucially this does NOT raise an
    alarm anywhere in the system: the data simply stops arriving, and it is up
    to the agent to notice the gap and troubleshoot it.
    """
    gc = cfg.glucose

    if not patient.is_enrolled:
        patient.last_cgm_value = None
        return None

    # Each sensor carries its own fixed calibration bias, drawn once at
    # insertion from the sensor stream.
    if patient.sensor_bias is None:
        patient.sensor_bias = rng.gauss(0.0, gc.cgm_bias_sd)

    # Sensor reliability transitions.
    if patient.signal_lost:
        patient.signal_loss_steps_left -= 1
        if patient.signal_loss_steps_left <= 0:
            patient.signal_lost = False
    else:
        if rng.random() < gc.sensor_signal_loss_prob:
            patient.signal_lost = True
            patient.signal_loss_steps_left = rng.randint(*gc.signal_loss_steps)
        elif not patient.sensor_degraded and rng.random() < gc.sensor_degrade_prob:
            patient.sensor_degraded = True

    if patient.signal_lost or not patient.on_ward:
        patient.steps_since_valid_cgm += 1
        patient.last_cgm_value = None
        return None

    # Interstitial lag: the sensor reports where glucose was a couple of steps
    # ago, which is exactly why a rapidly falling patient can be worse than the
    # dashboard suggests.
    history = patient.glucose_history
    idx = max(0, len(history) - 1 - gc.cgm_lag_steps)
    lagged = history[idx]

    noise_sd = gc.cgm_noise_sd * (gc.cgm_degraded_noise_multiplier if patient.sensor_degraded else 1.0)
    value = lagged + patient.sensor_bias + rng.gauss(0.0, noise_sd)

    # Transient artefact - the main source of false alarms.
    if rng.random() < gc.cgm_spike_prob:
        value += rng.choice((-1.0, 1.0)) * gc.cgm_spike_magnitude * rng.uniform(0.6, 1.0)

    value = _clamp(value, gc.min_glucose, gc.max_glucose)
    patient.last_cgm_value = value
    patient.steps_since_valid_cgm = 0
    patient.cgm_history.append(value)
    if len(patient.cgm_history) > 32:
        del patient.cgm_history[:-32]
    return value


def poc_glucose(patient: PatientState, rng: random.Random, cfg) -> float:
    """Point-of-care capillary reading: close to truth, trusted over CGM."""
    gc = cfg.glucose
    return _clamp(
        patient.true_glucose + rng.gauss(0.0, gc.poc_noise_sd),
        gc.min_glucose,
        gc.max_glucose,
    )
