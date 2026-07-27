"""The simulation engine: one 5-minute tick of a 12-hour ward shift.

Order of operations within ``step()``:

  1. resolve the agent's action (movement or interaction)
  2. advance latent patient physiology
  3. advance the CGM sensing chain and raise/clear alarms
  4. advance staff availability and bed flow
  5. score the resulting state (per-step rewards and penalties)
  6. test the termination conditions

Keeping action resolution before physiology means an agent that treats a
hypoglycaemic patient this step gets credit before the next glucose sample -
which is what "rapid response" has to mean at a 5-minute resolution.

ACADEMIC MODEL ONLY - simplified placeholders, not clinical guidance.
Stdlib only (ships to the browser build).
"""

import random

from ..config import SimConfig
from .actions import (
    ADMIN_ACTIONS,
    Action,
    MOVEMENT_DELTAS,
    PATIENT_TARGETED_ACTIONS,
    ROLE_ACTIONS,
)
from .alarms import Alarm, AlarmKind, alarm_family, evaluate_alarms, is_false_alarm
from .bedflow import WardFlow
from .eligibility import (
    can_enrol,
    evaluate_eligibility,
    hard_exclusions,
    is_eligible_pre_consent,
    should_deenrol,
)
from .glucose import apply_treatment, poc_glucose, step_sensor, step_true_glucose
from .observations import build_observation, observation_size
from .patient import (
    DischargeStage,
    EnrolmentStatus,
    Location,
    PatientState,
    Specialty,
)
from .rewards import RewardTracker
from .staff import StaffPool
from .ward_map import WardMap


class WardEngine:
    """Stateful simulation of one ward shift."""

    def __init__(self, cfg: SimConfig | None = None, seed: int | None = None):
        self.cfg = cfg or SimConfig()
        self.reset(seed if seed is not None else self.cfg.seed)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> list[float]:
        # Separate random streams so that the telemetry-on and telemetry-off
        # arms are genuinely matched. If the sensor chain drew from the same
        # stream as the patients, switching telemetry off would shift every
        # subsequent draw and the two arms would see different wards - which
        # would make the whole comparison meaningless.
        # Common random numbers, so the telemetry and counterfactual arms
        # simulate the *same ward*. Streams are partitioned by scope:
        #   engine.rng         - ward level: patient sampling, arrivals, staff.
        #                        Consumes identically in both arms.
        #   patient.rng        - that patient's physiology, transfers, discharge
        #   patient.rng_sensor - that patient's CGM chain
        #   patient.rng_care   - routine monitoring checks on that patient
        #   patient.rng_action - consequences of the agent acting on them
        # Per-patient streams are the load-bearing part: an intervention on one
        # patient changes how many draws *they* consume, but cannot shift any
        # other patient's trajectory. With one shared stream, supporting a
        # single discharge would silently rewrite the rest of the ward.
        #
        # Note what this does NOT do: staff availability and bed capacity are
        # genuinely shared, so treating one patient can still delay another
        # through contention. That is a real ward effect and is meant to be
        # there. What is eliminated is *spurious* coupling through randomness.
        base = seed if seed is not None else 0
        self.rng = random.Random(seed)
        self.step_index = 0
        self.ward_map = WardMap(self.cfg.ward.n_beds)
        self.flow = WardFlow(self.rng, self.cfg, stream_seed=base)
        self.staff = StaffPool(self.rng, self.cfg)
        self.rewards = RewardTracker(self.cfg)

        self.agent_x, self.agent_y = self.ward_map.agent_start
        self.active_alarms: dict[int, Alarm] = {}
        self.alarm_log: list[Alarm] = []
        self.last_alarm_step: dict[tuple[int, AlarmKind], int] = {}
        self.dashboard_seen_step: int | None = None
        # bed -> (patient_id, cgm value, staleness) at the last board read.
        self.dashboard_snapshot: dict[int, tuple[int, float | None, int]] = {}
        self.dashboard_alarm_ids: set[int] = set()
        self.event_log: list[str] = []
        self._discharged_seen = 0  # how far through flow.discharged we've processed
        self.terminated = False
        self.truncated = False
        self.termination_reason: str | None = None

        # KPI counters
        self.kpi = {
            "steps": 0,
            "time_below_range_steps": 0,
            "severe_hypo_events": 0,
            "severe_hypo_missed": 0,
            "serious_adverse_events": 0,
            "alarms_raised": 0,
            "false_alarms_raised": 0,
            "alarms_acknowledged": 0,
            "alarm_response_steps_total": 0,
            "unconfirmed_alarm_steps": 0,
            "poc_tests": 0,
            "treatments": 0,
            "treatments_without_poc": 0,
            "unnecessary_treatments": 0,
            "escalations": 0,
            "escalations_successful": 0,
            "correct_enrolments": 0,
            "incorrect_enrolments": 0,
            "missed_eligible": 0,
            "correct_deenrolments": 0,
            "unnecessary_deenrolments": 0,
            "failed_deenrolment_steps": 0,
            "signal_loss_events": 0,
            "signal_loss_ignored_steps": 0,
            "sensor_troubleshoots": 0,
            "discharges": 0,
            "discharge_delay_steps": 0,
            "admissions": 0,
            "usual_care_detections": 0,
            "hypo_episodes": 0,
            "hypo_detected_by_telemetry": 0,
            "hypo_detected_by_usual_care": 0,
            "hypo_detection_delay_steps_total": 0,
            "hypo_detections": 0,
            # Restricted to the monitored cohort - the patients on telemetry,
            # or in the counterfactual arm the patients who would have been.
            # This is the like-for-like comparison; ward-wide figures are
            # diluted by the majority of patients who are never eligible.
            "cohort_hypo_episodes": 0,
            "cohort_hypo_detections": 0,
            "cohort_detection_delay_steps_total": 0,
            "cohort_time_below_range_steps": 0,
            "cohort_severe_hypo_events": 0,
            "max_queue_length": 0,
            "overcrowding_steps": 0,
            "invalid_actions": 0,
            "wrong_role_requests": 0,
            "staff_requests": 0,
        }
        self.last_action_result: str = "ready"
        return self.observation()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def observation(self) -> list[float]:
        return build_observation(self)

    @property
    def observation_size(self) -> int:
        return observation_size(self.cfg.ward.n_beds)

    def visible_alarms(self) -> list[Alarm]:
        """Alarms the agent can currently see.

        With telemetry off there is no dashboard at all. With telemetry on, the
        board is visible from the nurse station; away from it the agent sees
        only the alarms it has already looked at (modelling a glance at the
        board or a handheld, refreshed by CHECK_DASHBOARD).
        """
        if not self.cfg.telemetry_enabled:
            return []
        alarms = [a for a in self.active_alarms.values() if a.resolved_step is None]
        if self.ward_map.at_station(self.agent_x, self.agent_y):
            return alarms
        if self.dashboard_seen_step is None:
            return []
        # Compare against the identities captured at read time, not against the
        # step number. Actions resolve before alarms are generated, so an alarm
        # raised later in the very same step would satisfy
        # `raised_step <= dashboard_seen_step` and appear to have been seen
        # before it existed.
        return [a for a in alarms if id(a) in self.dashboard_alarm_ids]

    def _read_dashboard(self) -> None:
        """Take a snapshot of the telemetry board.

        The agent carries away what it saw, not a live feed. Everything the
        observation reports about glucose comes from this snapshot, so walking
        away and not coming back means working from stale numbers - which is
        the whole reason the board has a location.
        """
        self.dashboard_seen_step = self.step_index
        # Exactly which alarms were on the board at the moment of the read.
        self.dashboard_alarm_ids = {
            id(a) for a in self.active_alarms.values() if a.resolved_step is None
        }
        # Keyed by bed but stamped with patient identity: beds get reused, and
        # showing a new admission the previous occupant's glucose would be a
        # particularly nasty way to be wrong.
        self.dashboard_snapshot = {
            patient.bed: (
                patient.patient_id,
                patient.last_cgm_value,
                patient.steps_since_valid_cgm,
            )
            for patient in self.flow.patients()
            if patient.is_enrolled
        }

    def adjacent_bed(self) -> int | None:
        return self.ward_map.adjacent_bed(self.agent_x, self.agent_y)

    def adjacent_patient(self) -> PatientState | None:
        bed = self.adjacent_bed()
        if bed is None:
            return None
        return self.flow.patient_at_bed(bed)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict]:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        act = Action(int(action))
        self.rewards.begin_step()

        urgent_before = [a for a in self.active_alarms.values() if a.is_urgent and a.resolved_step is None]

        self._resolve_action(act)

        # Unsafe prioritisation: administrative work while somebody is alarming
        # urgently and unattended.
        if act in ADMIN_ACTIONS and urgent_before:
            unattended = [a for a in urgent_before if a.acknowledged_step is None]
            if unattended:
                self.rewards.unsafe_prioritisation()

        self._advance_world()
        self._score_state()
        self._check_termination()

        self.step_index += 1
        self.kpi["steps"] = self.step_index

        reward = self.rewards.end_step()
        return self.observation(), reward, self.terminated, self.truncated, self.info()

    # ------------------------------------------------------------------
    # Action resolution
    # ------------------------------------------------------------------
    def _resolve_action(self, act: Action) -> None:
        if act in MOVEMENT_DELTAS:
            dx, dy = MOVEMENT_DELTAS[act]
            nx, ny = self.agent_x + dx, self.agent_y + dy
            if self.ward_map.walkable(nx, ny):
                self.agent_x, self.agent_y = nx, ny
                self.last_action_result = "moved"
                if self.ward_map.at_station(nx, ny) and self.cfg.telemetry_enabled:
                    self._read_dashboard()
            else:
                self._invalid("blocked")
            return

        if act is Action.WAIT:
            self.last_action_result = "waited"
            return

        if act is Action.CHECK_DASHBOARD:
            if not self.cfg.telemetry_enabled:
                self._invalid("no_telemetry")
                return
            # Telemetry is pushed to a handheld as well as the central monitor,
            # so the agent can check from anywhere - but it costs a step, and
            # what it learns is a snapshot that then ages. Standing at the
            # nurse station refreshes it for free.
            self._read_dashboard()
            self.last_action_result = "checked dashboard"
            return

        if act is Action.PRIORITISE_BEDFLOW:
            before = self.flow.queue_length
            moved = self.flow.prioritise_bedflow(self.step_index)
            reduced = before - self.flow.queue_length
            if reduced > 0:
                self.rewards.queue_reduced(reduced)
            self.last_action_result = f"bed flow ({moved} moved)" if moved else "bed flow (nothing to do)"
            if moved == 0:
                self._invalid("nothing_to_prioritise", penalise=False)
            return

        if act in ROLE_ACTIONS:
            self._resolve_staff_request(act)
            return

        # Everything else needs a patient in an adjacent bed.
        if act in PATIENT_TARGETED_ACTIONS:
            patient = self.adjacent_patient()
            if patient is None:
                self._invalid("no_patient")
                return
            self._resolve_patient_action(act, patient)
            return

        self._invalid("unhandled")

    def _invalid(self, reason: str, penalise: bool = True) -> None:
        if penalise:
            self.rewards.invalid_action()
            self.kpi["invalid_actions"] += 1
        self.last_action_result = f"no effect ({reason})"

    # ------------------------------------------------------------------
    def _resolve_staff_request(self, act: Action) -> None:
        role = ROLE_ACTIONS[act]
        patient = self.adjacent_patient()
        task = self._infer_task(patient)
        ok, outcome = self.staff.request(role, task)
        self.kpi["staff_requests"] += 1

        if outcome == "wrong_role":
            self.rewards.wrong_role_request()
            self.kpi["wrong_role_requests"] += 1
            self.last_action_result = f"{role} cannot help with {task}"
            return
        if not ok:
            self.last_action_result = f"{role} busy"
            return

        # A colleague who helps actually does something useful.
        self.last_action_result = f"{role} helped with {task}"
        if patient is None:
            return
        if task in ("check", "poc"):
            self._do_poc_test(patient, by_colleague=True)
        elif task == "treat":
            self._maybe_treat(patient, colleague=True)
        elif task in ("discharge_prep", "discharge_decision", "surgical_discharge_decision"):
            if patient.discharge_stage is DischargeStage.READY:
                patient.discharge_stage = DischargeStage.REVIEWED
                patient.knowledge.discharge_reviewed_step = self.step_index
                patient.knowledge.known_discharge_ready = True
            elif patient.discharge_stage is DischargeStage.REVIEWED:
                if self.flow.support_discharge(patient):
                    self.rewards.discharge_supported()
        elif task == "review":
            self._review_notes(patient)

    def _infer_task(self, patient: PatientState | None) -> str:
        """What the agent is implicitly asking for help with, given context.

        The *task* comes from the situation; choosing the right *role* for it is
        the agent's job, and getting it wrong wastes the step.
        """
        if patient is None:
            return "bedflow"
        alarm = self.active_alarms.get(patient.bed)
        if alarm is not None and alarm.resolved_step is None:
            if alarm.poc_confirmed_step is None:
                return "poc"
            return "treat"
        if patient.discharge_stage is DischargeStage.READY:
            return (
                "surgical_discharge_decision"
                if patient.specialty is Specialty.SURGICAL
                else "discharge_decision"
            )
        if patient.discharge_stage is DischargeStage.REVIEWED:
            return "discharge_prep"
        if patient.is_enrolled:
            return "review"
        return "check"

    # ------------------------------------------------------------------
    def _resolve_patient_action(self, act: Action, patient: PatientState) -> None:
        if act is Action.CHECK_PATIENT:
            self._check_patient(patient)
        elif act is Action.REVIEW_NOTES:
            self._review_notes(patient)
        elif act is Action.ASK_CONSENT:
            self._ask_consent(patient)
        elif act is Action.ENROL:
            self._enrol(patient)
        elif act is Action.REVIEW_ELIGIBILITY:
            self._review_eligibility(patient)
        elif act is Action.DEENROL:
            self._deenrol(patient)
        elif act is Action.RESPOND_ALARM:
            self._respond_alarm(patient)
        elif act is Action.POC_GLUCOSE_TEST:
            self._do_poc_test(patient)
        elif act is Action.TREAT_HYPO:
            self._treat(patient, "hypo")
        elif act is Action.TREAT_HYPER:
            self._treat(patient, "hyper")
        elif act is Action.ESCALATE:
            self._escalate(patient)
        elif act is Action.TROUBLESHOOT_SENSOR:
            self._troubleshoot(patient)
        elif act is Action.SUPPORT_DISCHARGE:
            self._support_discharge(patient)

    # --- information gathering ----------------------------------------
    def _check_patient(self, patient: PatientState) -> None:
        if not patient.visible_at_bed:
            self._invalid("patient_not_at_bed")
            return
        k = patient.knowledge
        k.last_checked_step = self.step_index

        # Looking at a patient is a real discovery route, and the only one the
        # agent has when telemetry is off. A hypoglycaemic patient at the
        # bedside is often visibly unwell - sweating, drowsy, confused - and
        # more obviously so the lower they are.
        ac = self.cfg.alarms
        uc = self.cfg.usual_care
        if patient.true_glucose < ac.hypo_threshold:
            # Drawn from the ACTION stream. This is a consequence of the agent
            # choosing to look, so putting it on rng_care would let an action
            # shift the exogenous routine-monitoring sequence and desynchronise
            # the two arms.
            noticed = (
                patient.true_glucose < ac.severe_hypo_threshold
                or patient.rng_action.random() < uc.bedside_symptom_recognition
            )
            if noticed:
                self._record_hypo_detection(patient, "usual_care")
                self.last_action_result = (
                    f"bed {patient.bed} looks unwell - check their glucose"
                )
                return
        # A bedside check reveals obvious discharge readiness and symptoms, and
        # in a no-telemetry world it is the ONLY way to notice a low patient.
        k.discharge_reviewed_step = self.step_index
        k.known_discharge_ready = patient.discharge_stage in (
            DischargeStage.READY,
            DischargeStage.REVIEWED,
            DischargeStage.SUPPORTED,
        )
        self.last_action_result = f"checked bed {patient.bed}"

    def _review_notes(self, patient: PatientState) -> None:
        k = patient.knowledge
        k.eligibility_reviewed_step = self.step_index
        k.known_has_diabetes = patient.has_diabetes
        k.known_two_or_more_injections = patient.two_or_more_injections
        k.known_expected_los_at_least_48h = patient.expected_los_at_least_48h
        k.known_has_capacity = patient.has_capacity
        k.known_excluded = patient.pregnant_or_breastfeeding or patient.end_of_life
        self.last_action_result = f"reviewed notes for bed {patient.bed}"

        # Correctly identifying an ineligible patient is worth something: it is
        # the work that prevents an inappropriate enrolment.
        if hard_exclusions(patient) and not patient.is_enrolled:
            self.rewards.correct_ineligible_identification()

    def _ask_consent(self, patient: PatientState) -> None:
        if not patient.visible_at_bed:
            self._invalid("patient_not_at_bed")
            return
        if patient.consent_asked:
            self._invalid("already_asked")
            return
        if not patient.has_capacity:
            # Asking somebody who lacks capacity is not informed consent.
            patient.consent_asked = True
            patient.consent_declined = True
            patient.knowledge.consent_asked_step = self.step_index
            patient.knowledge.known_consented = False
            patient.knowledge.known_has_capacity = False
            self.last_action_result = f"bed {patient.bed} lacks capacity to consent"
            return

        patient.consent_asked = True
        patient.knowledge.consent_asked_step = self.step_index
        if patient.will_consent:
            patient.knowledge.known_consented = True
            self.last_action_result = f"bed {patient.bed} consented"
        else:
            patient.consent_declined = True
            patient.enrolment = EnrolmentStatus.DECLINED
            patient.knowledge.known_consented = False
            self.last_action_result = f"bed {patient.bed} declined"

    # --- enrolment ----------------------------------------------------
    def _enrol(self, patient: PatientState) -> None:
        if not self.cfg.telemetry_enabled:
            # There is no telemetry programme to enrol into in the counterfactual
            # arm, so the whole enrolment pathway is unavailable there.
            self._invalid("no_telemetry")
            return
        ok, reason = can_enrol(patient)
        if not ok:
            self._invalid(reason)
            return
        result = evaluate_eligibility(patient)
        patient.enrolment = EnrolmentStatus.ENROLLED
        patient.enrolled_step = self.step_index
        # NOTE: enrolling does NOT add the patient to `telemetry_cohort`. That
        # cohort is fixed at handover and is identical in both arms by
        # construction, which is what makes the cohort-restricted outcome a
        # like-for-like comparison. Mid-shift enrolments happen only in the
        # telemetry arm, so counting them would make the two cohorts different
        # populations and the primary estimand incomparable. They are still
        # captured by the ward-wide metrics and by enrolment precision/recall.
        patient.sensor_bias = None  # fresh sensor; bias drawn on first reading
        patient.steps_since_valid_cgm = 0
        if result.eligible:
            self.rewards.correct_enrolment()
            self.kpi["correct_enrolments"] += 1
            self.last_action_result = f"enrolled bed {patient.bed}"
        else:
            self.rewards.enrolled_ineligible()
            self.kpi["incorrect_enrolments"] += 1
            self.last_action_result = f"enrolled INELIGIBLE bed {patient.bed} ({', '.join(result.reasons)})"

    def _review_eligibility(self, patient: PatientState) -> None:
        if not patient.is_enrolled:
            self._invalid("not_enrolled")
            return
        self._review_notes(patient)
        needs, reasons = should_deenrol(patient)
        self.last_action_result = (
            f"bed {patient.bed} now ineligible ({', '.join(reasons)})"
            if needs
            else f"bed {patient.bed} still eligible"
        )

    def _deenrol(self, patient: PatientState) -> None:
        if not patient.is_enrolled:
            self._invalid("not_enrolled")
            return
        needs, reasons = should_deenrol(patient)
        patient.enrolment = EnrolmentStatus.DEENROLLED
        patient.last_cgm_value = None
        patient.cgm_history.clear()
        self._clear_alarm(patient.bed, "deenrolled")
        if needs:
            self.rewards.correct_deenrolment()
            self.kpi["correct_deenrolments"] += 1
            self.last_action_result = f"de-enrolled bed {patient.bed} ({', '.join(reasons)})"
        else:
            self.rewards.unnecessary_deenrolment()
            self.kpi["unnecessary_deenrolments"] += 1
            self.last_action_result = f"unnecessary de-enrolment, bed {patient.bed}"

    # --- clinical response --------------------------------------------
    def _respond_alarm(self, patient: PatientState) -> None:
        alarm = self.active_alarms.get(patient.bed)
        if alarm is None or alarm.resolved_step is not None:
            self._invalid("no_alarm")
            return
        if alarm.acknowledged_step is None:
            alarm.acknowledged_step = self.step_index
            self.kpi["alarms_acknowledged"] += 1
            response = alarm.response_time() or 0
            self.kpi["alarm_response_steps_total"] += response
            if response <= self.cfg.alarms.response_deadline_steps:
                # Faster responses earn proportionally more.
                self.rewards.fast_alarm_response(
                    1.0 - response / (self.cfg.alarms.response_deadline_steps + 1)
                )
        self.last_action_result = f"responded to {alarm.kind.value} at bed {patient.bed}"

    def _do_poc_test(self, patient: PatientState, by_colleague: bool = False) -> None:
        if not patient.visible_at_bed:
            self._invalid("patient_not_at_bed")
            return
        value = poc_glucose(patient, patient.rng_action, self.cfg)
        patient.knowledge.last_poc_step = self.step_index
        patient.knowledge.last_poc_value = value
        self.kpi["poc_tests"] += 1
        alarm = self.active_alarms.get(patient.bed)
        if alarm is not None and alarm.resolved_step is None:
            alarm.poc_confirmed_step = self.step_index
            alarm.poc_value = value
            if alarm.acknowledged_step is None:
                alarm.acknowledged_step = self.step_index
                self.kpi["alarms_acknowledged"] += 1
                self.kpi["alarm_response_steps_total"] += alarm.response_time() or 0
        who = "colleague" if by_colleague else "you"
        self.last_action_result = f"PoC bed {patient.bed}: {value:.1f} mmol/L ({who})"

    def _treat(self, patient: PatientState, kind: str) -> None:
        if not patient.visible_at_bed:
            self._invalid("patient_not_at_bed")
            return
        self._maybe_treat(patient, kind=kind)

    def _maybe_treat(self, patient: PatientState, kind: str | None = None, colleague: bool = False) -> None:
        ac = self.cfg.alarms
        k = patient.knowledge
        alarm = self.active_alarms.get(patient.bed)

        # Which condition is the agent treating?
        if kind is None:
            if patient.true_glucose < ac.hypo_threshold:
                kind = "hypo"
            else:
                kind = "hyper"

        poc_fresh = (
            k.last_poc_step is not None
            and self.step_index - k.last_poc_step <= self.cfg.alarms.response_deadline_steps
        )
        poc_value = k.last_poc_value if poc_fresh else None

        # PoC is trusted over CGM. Treating without confirming a clinically
        # significant alarm is penalised; treating when PoC says the patient is
        # fine is penalised harder.
        if alarm is not None and alarm.needs_poc and not poc_fresh:
            self.rewards.treatment_without_poc()
            self.kpi["treatments_without_poc"] += 1
            patient.treated_without_poc = True
        elif alarm is None and not poc_fresh:
            self.rewards.treatment_without_poc()
            self.kpi["treatments_without_poc"] += 1

        truly_hypo = patient.true_glucose < ac.hypo_threshold
        truly_hyper = patient.true_glucose > patient.hyper_threshold(ac.hyper_threshold_default)
        reference = poc_value if poc_value is not None else patient.true_glucose

        wrong_direction = (kind == "hypo" and reference >= ac.hypo_threshold) or (
            kind == "hyper" and reference <= patient.hyper_threshold(ac.hyper_threshold_default)
        )

        if wrong_direction:
            # Either a false alarm that was not confirmed, or the wrong patient.
            self.rewards.unnecessary_treatment()
            self.kpi["unnecessary_treatments"] += 1
            if not (truly_hypo or truly_hyper):
                self.rewards.wrong_patient_treatment()
            self.last_action_result = f"unnecessary {kind} treatment at bed {patient.bed}"
            self._clear_alarm(patient.bed, "treated")
            return

        apply_treatment(patient, kind, self.cfg)
        patient.last_treatment_step = self.step_index
        self.kpi["treatments"] += 1

        if kind == "hypo" and truly_hypo:
            severity = 2.0 if patient.true_glucose < ac.severe_hypo_threshold else 1.0
            if alarm is not None and alarm.age(self.step_index) <= self.cfg.alarms.response_deadline_steps:
                self.rewards.hypo_treated_promptly(severity)
            elif alarm is None:
                # Caught it without an alarm at all - the pre-emptive case.
                self.rewards.hypo_prevented()
            else:
                self.rewards.hypo_treated_promptly(severity * 0.5)
            patient.severe_hypo_untreated_steps = 0

        self._clear_alarm(patient.bed, "treated")
        who = "colleague" if colleague else "you"
        self.last_action_result = f"treated {kind} at bed {patient.bed} ({who})"

    def _escalate(self, patient: PatientState) -> None:
        ac = self.cfg.alarms
        self.kpi["escalations"] += 1
        ok, role = self.staff.escalate()
        severe = patient.true_glucose < ac.severe_hypo_threshold
        recurrent = self._is_recurrent(patient)

        if not ok:
            self.last_action_result = "escalation: nobody available"
            return

        self.kpi["escalations_successful"] += 1
        if severe or recurrent:
            self.rewards.correct_escalation()
            apply_treatment(patient, "hypo" if severe else "hyper", self.cfg, escalated=True)
            patient.last_treatment_step = self.step_index
            patient.severe_hypo_untreated_steps = 0
            self._clear_alarm(patient.bed, "escalated")
            self.last_action_result = f"escalated bed {patient.bed} to {role}"
        else:
            # Escalating a stable patient wastes a colleague's time.
            self.last_action_result = f"escalated bed {patient.bed} to {role} (not indicated)"

    def _is_recurrent(self, patient: PatientState) -> bool:
        """Two or more alarms for this patient during the shift."""
        return sum(1 for a in self.alarm_log if a.bed == patient.bed) >= 2

    def _troubleshoot(self, patient: PatientState) -> None:
        if not patient.is_enrolled:
            self._invalid("not_enrolled")
            return
        if not (patient.signal_lost or patient.sensor_degraded):
            self._invalid("sensor_fine")
            return
        self.kpi["sensor_troubleshoots"] += 1
        if patient.rng_action.random() < self.cfg.glucose.troubleshoot_success_prob:
            patient.signal_lost = False
            patient.signal_loss_steps_left = 0
            patient.sensor_degraded = False
            patient.steps_since_valid_cgm = 0
            self.last_action_result = f"sensor restored at bed {patient.bed}"
        else:
            self.last_action_result = f"sensor still faulty at bed {patient.bed}"

    def _support_discharge(self, patient: PatientState) -> None:
        if patient.discharge_stage is DischargeStage.READY:
            patient.discharge_stage = DischargeStage.REVIEWED
            patient.knowledge.discharge_reviewed_step = self.step_index
            patient.knowledge.known_discharge_ready = True
            self.last_action_result = f"reviewed discharge for bed {patient.bed}"
            return
        if patient.discharge_stage is DischargeStage.REVIEWED:
            if self.flow.support_discharge(patient):
                self.rewards.discharge_supported()
                self.last_action_result = f"supporting discharge, bed {patient.bed}"
                return
        self._invalid("not_ready_for_discharge")

    # ------------------------------------------------------------------
    # World advancement
    # ------------------------------------------------------------------
    def _advance_world(self) -> None:
        pc = self.cfg.patients

        for patient in list(self.flow.patients()):
            # Everything that happens *to* a patient is drawn from that
            # patient's own streams, so intervening on one cannot perturb any
            # other patient's trajectory.
            step_true_glucose(patient, patient.rng, self.cfg)
            self._maybe_change_eligibility(patient, pc)
            was_lost = patient.signal_lost
            cgm_value = step_sensor(patient, patient.rng_sensor, self.cfg)
            if patient.signal_lost and not was_lost:
                self.kpi["signal_loss_events"] += 1
            self._update_alarms(patient, cgm_value)
            self._track_hypo_episode(patient)
            self._usual_care_monitoring(patient)

        self.staff.step()
        events = self.flow.step(self.step_index)
        self.kpi["admissions"] += events["admitted"]
        self.kpi["discharges"] = self.flow.total_discharges
        self.kpi["max_queue_length"] = max(self.kpi["max_queue_length"], self.flow.queue_length)

        # Patients who leave take their enrolment and alarms with them. Only
        # the newly discharged are processed: iterating the cumulative list
        # would keep clearing alarms on their old bed, silencing whoever is
        # admitted into it next.
        newly_discharged = self.flow.discharged[self._discharged_seen:]
        self._discharged_seen = len(self.flow.discharged)
        for patient in newly_discharged:
            if self.active_alarms.get(patient.bed) is not None:
                self._clear_alarm(patient.bed, "discharged")
            if not patient.counted_missed_eligible:
                patient.counted_missed_eligible = True
                if not patient.is_enrolled and is_eligible_pre_consent(patient):
                    self.rewards.missed_eligible_patient()
                    self.kpi["missed_eligible"] += 1

    def _track_hypo_episode(self, patient: PatientState) -> None:
        """Record when a hypoglycaemic episode starts and how it was found.

        Detection latency - how long a patient spends below range before
        anybody knows - is the primary outcome the simulator exists to compare
        between telemetry and routine monitoring.
        """
        ac = self.cfg.alarms
        below = patient.true_glucose < ac.hypo_threshold

        if below:
            patient.hypo_recovery_steps = 0
            patient.hypo_consecutive_low_steps += 1
            if patient.hypo_episode_started_step is None:
                patient.hypo_episode_started_step = self.step_index
                patient.hypo_episode_detected = False
                patient.hypo_episode_counted = False
            # Qualification counts CONSECUTIVE below-threshold samples, not
            # elapsed time since onset. Using elapsed time would let a trace
            # like 3.5, 4.0, 4.0, 3.5 - two low readings, never fifteen
            # continuous minutes - qualify as an event.
            duration = patient.hypo_consecutive_low_steps
            if not patient.hypo_episode_counted and duration >= ac.hypo_event_min_steps:
                patient.hypo_episode_counted = True
                self.kpi["hypo_episodes"] += 1
                if patient.telemetry_cohort:
                    self.kpi["cohort_hypo_episodes"] += 1
                # Telemetry usually detects well before the 15-minute mark, so
                # a detection already banked is credited here.
                if patient.hypo_episode_detected:
                    self._bank_hypo_detection(patient)
        else:
            # An episode ends only after SUSTAINED recovery. A single reading
            # back above threshold does not end it - otherwise a patient
            # hovering at 3.5, 4.0, 3.5 is counted as two separate events when
            # clinically it is plainly one. The consecutive-low run breaks
            # immediately, though: qualification needs unbroken time below.
            patient.hypo_recovery_steps += 1
            patient.hypo_consecutive_low_steps = 0
            if patient.hypo_recovery_steps >= ac.hypo_recovery_min_steps:
                patient.hypo_episode_started_step = None
                patient.hypo_episode_detected = False
                patient.hypo_episode_counted = False
                patient.hypo_episode_detected_step = None
                patient.hypo_episode_detected_route = None
                patient.hypo_recovery_steps = 0
            return

        if not below or patient.hypo_episode_detected:
            return

        # Telemetry counts as detection only when the alarm has actually been
        # SEEN. An alarm sitting on an unread board is not detection: nobody
        # knows. Measuring from alarm generation instead would report device
        # latency while claiming to report "time before anybody knows".
        alarm = self.active_alarms.get(patient.bed)
        if (
            self.cfg.telemetry_enabled
            and alarm is not None
            and alarm.resolved_step is None
            and alarm.kind in (AlarmKind.HYPO, AlarmKind.SEVERE_HYPO)
            and any(a is alarm for a in self.visible_alarms())
        ):
            self._record_hypo_detection(patient, "telemetry")

    def _record_hypo_detection(self, patient: PatientState, route: str) -> None:
        """Note that somebody has found this episode, and by which route."""
        if patient.hypo_episode_detected:
            return
        patient.hypo_episode_detected = True
        patient.hypo_episode_detected_step = self.step_index
        patient.hypo_episode_detected_route = route
        # Only episodes that meet the 15-minute event definition contribute to
        # the detection statistics; a dip that resolves before then was never
        # an event to detect.
        if patient.hypo_episode_counted:
            self._bank_hypo_detection(patient)

    def _bank_hypo_detection(self, patient: PatientState) -> None:
        started = patient.hypo_episode_started_step
        detected = patient.hypo_episode_detected_step
        if started is None or detected is None:
            return
        delay = detected - started
        self.kpi["hypo_detection_delay_steps_total"] += delay
        self.kpi["hypo_detections"] += 1
        if patient.telemetry_cohort:
            self.kpi["cohort_detection_delay_steps_total"] += delay
            self.kpi["cohort_hypo_detections"] += 1
        if patient.hypo_episode_detected_route == "telemetry":
            self.kpi["hypo_detected_by_telemetry"] += 1
        else:
            self.kpi["hypo_detected_by_usual_care"] += 1

    def _usual_care_monitoring(self, patient: PatientState) -> None:
        """Routine capillary rounds and symptom recognition by ward staff.

        This is the comparator, and it applies to EVERY patient - telemetry is
        additive to standard ward care, not a substitute for it. A patient on
        CGM still gets their routine checks; the question the simulator asks is
        whether the alarm gets there first, and by how much.
        """
        uc = self.cfg.usual_care
        ac = self.cfg.alarms

        if patient.true_glucose >= ac.hypo_threshold or not patient.on_ward:
            return

        prob = uc.routine_detection_prob
        if patient.true_glucose < ac.severe_hypo_threshold:
            prob *= uc.severe_detection_multiplier
        if self.staff.coarse_availability() == 0:
            prob *= uc.understaffed_multiplier

        if patient.rng_care.random() >= prob:
            return

        self.kpi["usual_care_detections"] += 1
        if not patient.hypo_episode_detected:
            self._record_hypo_detection(patient, "usual_care")

        if uc.treat_on_detection:
            # Background staff treat it. No reward for the agent: this is the
            # baseline standard of care, not the agent's achievement.
            apply_treatment(patient, "hypo", self.cfg)
            patient.last_treatment_step = self.step_index
            patient.severe_hypo_untreated_steps = 0
            self.event_log.append(
                f"bed {patient.bed}: hypoglycaemia found on routine monitoring"
            )

    def _maybe_change_eligibility(self, patient: PatientState, pc) -> None:
        """Mid-shift changes that can make an enrolled patient ineligible.

        Both draws happen unconditionally. A patient's insulin regimen changes
        because of their clinical course, not because somebody put a sensor on
        them, so gating the draw on enrolment would make the exogenous stream
        diverge between the telemetry and counterfactual arms - and the matched
        comparison would quietly stop being matched.
        """
        regimen_draw = patient.rng.random()
        if (
            patient.two_or_more_injections
            and patient.has_diabetes
            and regimen_draw < pc.regimen_reduction_prob
        ):
            patient.insulin_injections_per_day = 1
            if patient.is_enrolled:
                patient.became_ineligible_step = self.step_index
            self.event_log.append(f"bed {patient.bed}: insulin reduced to once daily")
        eol_draw = patient.rng.random()
        if not patient.end_of_life and eol_draw < pc.prob_becomes_end_of_life:
            patient.end_of_life = True
            if patient.is_enrolled:
                patient.became_ineligible_step = self.step_index
                self.event_log.append(f"bed {patient.bed}: transitioned to end-of-life care")

        # A revised discharge plan can bring the expected stay under 48 hours,
        # which is one of the documented ways a patient becomes ineligible.
        plan_draw = patient.rng.random()
        if plan_draw < pc.prob_discharge_plan_revised:
            patient.expected_los_hours = min(
                patient.expected_los_hours,
                patient.steps_on_ward * self.cfg.minutes_per_step / 60.0 + 24.0,
            )
            if patient.is_enrolled:
                patient.became_ineligible_step = self.step_index
                self.event_log.append(f"bed {patient.bed}: discharge brought forward")

        # A patient may withdraw consent at any point. Drawn unconditionally:
        # gating on `is_enrolled` would make the draw telemetry-dependent and
        # desynchronise the two arms (caught by test_counterfactual_rng).
        withdraw_draw = patient.rng.random()
        if patient.consent_asked and withdraw_draw < pc.prob_withdraws_consent:
            patient.consent_declined = True
            patient.became_ineligible_step = self.step_index
            self.event_log.append(f"bed {patient.bed}: withdrew consent")

    def _update_alarms(self, patient: PatientState, cgm_value: float | None) -> None:
        if not self.cfg.telemetry_enabled:
            return

        # A patient who has left the ward for theatre or imaging is out of
        # sensor range and cannot be assessed at the bedside. Holding a live
        # alarm against an empty bed would only strand the agent there.
        if not patient.on_ward:
            patient.alarm_streak = {}
            if patient.bed in self.active_alarms:
                self._clear_alarm(patient.bed, "patient_off_ward")
            return
        kinds = evaluate_alarms(patient, cgm_value, self.step_index, self.cfg)
        existing = self.active_alarms.get(patient.bed)

        if not kinds:
            patient.alarm_streak = {}
            # Clear an alarm once the displayed value comes back into range.
            if existing is not None and existing.resolved_step is None and cgm_value is not None:
                self._clear_alarm(patient.bed, "auto_resolved")
            return

        # Highest-severity alarm wins the dashboard slot.
        priority = [
            AlarmKind.SEVERE_HYPO,
            AlarmKind.HYPO,
            AlarmKind.RAPID_FALL,
            AlarmKind.HYPER,
            AlarmKind.RAPID_RISE,
        ]
        kind = next(k for k in priority if k in kinds)

        # Persistence logic: an out-of-range reading has to repeat before it
        # raises an alarm, which suppresses single-sample artefacts. Tracked by
        # clinical family so that a patient oscillating across the severe
        # threshold still accumulates a streak; the severity actually reported
        # is whatever the latest reading shows.
        needed = self.cfg.alarms.persistence_readings
        family = alarm_family(kind)
        if needed > 1:
            streak = patient.alarm_streak.get(family, 0) + 1
            patient.alarm_streak = {family: streak}
            if streak < needed:
                return
        else:
            patient.alarm_streak = {}

        if existing is not None and existing.resolved_step is None:
            if existing.kind is kind:
                return
            # Escalating severity replaces the alarm.
            if priority.index(kind) < priority.index(existing.kind):
                self._clear_alarm(patient.bed, "superseded")
            else:
                return

        last = self.last_alarm_step.get((patient.bed, kind))
        if last is not None and self.step_index - last < self.cfg.alarms.realarm_cooldown_steps:
            return

        alarm = Alarm(
            bed=patient.bed,
            kind=kind,
            raised_step=self.step_index,
            cgm_value=cgm_value if cgm_value is not None else 0.0,
            false_alarm=is_false_alarm(patient, kind, self.cfg),
        )
        self.active_alarms[patient.bed] = alarm
        self.alarm_log.append(alarm)
        self.last_alarm_step[(patient.bed, kind)] = self.step_index
        self.kpi["alarms_raised"] += 1
        if alarm.false_alarm:
            self.kpi["false_alarms_raised"] += 1

    def _clear_alarm(self, bed: int, reason: str) -> None:
        alarm = self.active_alarms.get(bed)
        if alarm is None:
            return
        alarm.resolved_step = self.step_index
        del self.active_alarms[bed]

    # ------------------------------------------------------------------
    # Per-step scoring
    # ------------------------------------------------------------------
    def _score_state(self) -> None:
        ac = self.cfg.alarms
        below_range = 0
        unattended_alarms = 0
        ignored_signal_loss = 0
        failure_to_deenrol = 0
        discharge_delays = 0
        safe_enrolled = 0

        for patient in self.flow.patients():
            # Time below range, whether or not anyone noticed.
            if patient.true_glucose < ac.hypo_threshold:
                below_range += 1
                self.kpi["time_below_range_steps"] += 1
                if patient.telemetry_cohort:
                    self.kpi["cohort_time_below_range_steps"] += 1

            # Severe hypoglycaemia left untreated becomes a serious event.
            # A hysteresis band stops a patient oscillating around 3.0 mmol/L
            # from being counted as a fresh event every few steps.
            if patient.true_glucose >= ac.severe_hypo_threshold + ac.false_alarm_margin:
                patient.severe_episode_active = False

            if patient.true_glucose < ac.severe_hypo_threshold:
                patient.severe_hypo_untreated_steps += 1
                if not patient.severe_episode_active:
                    patient.severe_episode_active = True
                    self.kpi["severe_hypo_events"] += 1
                    if patient.telemetry_cohort:
                        self.kpi["cohort_severe_hypo_events"] += 1
                sae_steps = self.cfg.usual_care.sae_untreated_steps
                if patient.severe_hypo_untreated_steps == sae_steps and not patient.sae_recorded:
                    patient.sae_recorded = True
                    self.kpi["serious_adverse_events"] += 1
                    self.kpi["severe_hypo_missed"] += 1
                    self.rewards.missed_severe_hypo()
                    self.rewards.serious_adverse_event()
            else:
                patient.severe_hypo_untreated_steps = 0

            if patient.is_enrolled:
                # Signal loss the agent has not acted on.
                if patient.steps_since_valid_cgm > ac.signal_loss_grace_steps:
                    ignored_signal_loss += 1
                    self.kpi["signal_loss_ignored_steps"] += 1

                # Still enrolled despite no longer meeting the criteria.
                needs, _ = should_deenrol(patient)
                if needs:
                    failure_to_deenrol += 1
                    self.kpi["failed_deenrolment_steps"] += 1

                if patient.true_glucose >= ac.hypo_threshold and not needs:
                    safe_enrolled += 1

            if patient.discharge_stage in (DischargeStage.READY, DischargeStage.REVIEWED):
                discharge_delays += 1
                self.kpi["discharge_delay_steps"] += 1

        unconfirmed_alarms = 0
        for alarm in self.active_alarms.values():
            if alarm.resolved_step is not None:
                continue
            if (
                alarm.acknowledged_step is None
                and alarm.age(self.step_index) > ac.response_deadline_steps
            ):
                unattended_alarms += 1
            # Acknowledging an alarm is not the same as acting on it. A
            # clinically significant alarm that has been noticed but never
            # confirmed with a capillary test keeps accruing a penalty -
            # otherwise the agent could silence the board by acknowledging
            # everything and confirming nothing.
            if (
                alarm.needs_poc
                and alarm.poc_confirmed_step is None
                and alarm.age(self.step_index) > ac.response_deadline_steps
            ):
                unconfirmed_alarms += 1

        self.rewards.time_below_range(below_range)
        self.rewards.delayed_alarm_response(unattended_alarms)
        self.rewards.unconfirmed_significant_alarm(unconfirmed_alarms)
        self.kpi["unconfirmed_alarm_steps"] += unconfirmed_alarms
        self.rewards.ignored_signal_loss(ignored_signal_loss)
        self.rewards.failure_to_deenrol(failure_to_deenrol)
        self.rewards.discharge_delay(discharge_delays)
        self.rewards.enrolled_patients_safe(safe_enrolled)

        queue_length = self.flow.queue_length
        self.rewards.queue_pressure(queue_length)
        if self.flow.overcrowded:
            self.rewards.overcrowding()
            self.kpi["overcrowding_steps"] += 1
        if self.flow.safe_occupancy:
            self.rewards.safe_occupancy()
        if self.staff.overloaded:
            self.rewards.staff_overload()

    # ------------------------------------------------------------------
    def _check_termination(self) -> None:
        if self.cfg.terminate_on_unsafe_overcrowding and self.flow.unsafe_overcrowding:
            self.terminated = True
            self.termination_reason = "unsafe_overcrowding"
            self.rewards.unsafe_overcrowding()
            return

        if self.cfg.terminate_on_sae and self.kpi["serious_adverse_events"] > 0:
            self.terminated = True
            self.termination_reason = "serious_adverse_event"
            return

        if self.step_index + 1 >= self.cfg.steps_per_episode:
            self.truncated = True
            self.termination_reason = "shift_end"
            if self.kpi["serious_adverse_events"] == 0:
                self.rewards.shift_completed_without_sae()
            self._award_alarm_fatigue_bonus()
            self._count_missed_eligible_at_handover()

    def _count_missed_eligible_at_handover(self) -> None:
        """Penalise eligible patients who spent the whole shift un-enrolled.

        A patient who would have consented and met every criterion, and was on
        the ward all shift without being approached, is a missed opportunity -
        the recall half of the enrolment-quality metric.
        """
        if not self.cfg.telemetry_enabled:
            return  # no enrolment pathway exists in the counterfactual arm
        for patient in self.flow.patients():
            if patient.counted_missed_eligible or patient.is_enrolled:
                continue
            patient.counted_missed_eligible = True
            if is_eligible_pre_consent(patient):
                self.rewards.missed_eligible_patient()
                self.kpi["missed_eligible"] += 1

    def _award_alarm_fatigue_bonus(self) -> None:
        """Reward keeping the nuisance-alarm burden down over the shift."""
        total = self.kpi["alarms_raised"]
        if total == 0:
            self.rewards.alarm_fatigue_bonus(1.0)
            return
        nuisance = self.kpi["false_alarms_raised"]
        self.rewards.alarm_fatigue_bonus(1.0 - nuisance / total)

    # ------------------------------------------------------------------
    def info(self) -> dict:
        kpi = dict(self.kpi)
        acknowledged = kpi["alarms_acknowledged"]
        kpi["mean_alarm_response_steps"] = (
            kpi["alarm_response_steps_total"] / acknowledged if acknowledged else None
        )
        kpi["alarm_acknowledgement_rate"] = (
            acknowledged / kpi["alarms_raised"] if kpi["alarms_raised"] else None
        )
        kpi["false_alarm_rate"] = (
            kpi["false_alarms_raised"] / kpi["alarms_raised"] if kpi["alarms_raised"] else None
        )
        enrolments = kpi["correct_enrolments"] + kpi["incorrect_enrolments"]
        kpi["enrolment_precision"] = kpi["correct_enrolments"] / enrolments if enrolments else None
        recall_denominator = kpi["correct_enrolments"] + kpi["missed_eligible"]
        kpi["enrolment_recall"] = (
            kpi["correct_enrolments"] / recall_denominator if recall_denominator else None
        )
        # Detection latency: the primary outcome for the telemetry comparison.
        detections = kpi["hypo_detections"]
        kpi["mean_hypo_detection_delay_steps"] = (
            kpi["hypo_detection_delay_steps_total"] / detections if detections else None
        )
        kpi["hypo_detection_rate"] = (
            detections / kpi["hypo_episodes"] if kpi["hypo_episodes"] else None
        )
        # Cohort-restricted versions: the primary estimand. The monitored
        # cohort is fixed at handover and identical in both arms, so these are
        # the only like-for-like detection figures. Ward-wide numbers are
        # diluted by the ~86% of patients never eligible for telemetry.
        cohort_detections = kpi["cohort_hypo_detections"]
        kpi["cohort_mean_detection_delay_steps"] = (
            kpi["cohort_detection_delay_steps_total"] / cohort_detections
            if cohort_detections
            else None
        )
        kpi["cohort_detection_rate"] = (
            cohort_detections / kpi["cohort_hypo_episodes"]
            if kpi["cohort_hypo_episodes"]
            else None
        )
        kpi["enrolled_now"] = sum(1 for p in self.flow.patients() if p.is_enrolled)
        kpi["queue_length"] = self.flow.queue_length
        kpi["occupied_beds"] = self.flow.occupied_beds
        kpi["free_beds"] = self.flow.free_beds
        # The headline safety metric: a shift that ends with no serious incident.
        kpi["incident_free_shift"] = (
            self.kpi["serious_adverse_events"] == 0 if (self.terminated or self.truncated) else None
        )
        kpi["telemetry_enabled"] = self.cfg.telemetry_enabled

        return {
            "kpi": kpi,
            "reward_components": self.rewards.summary(),
            "termination_reason": self.termination_reason,
            "last_action_result": self.last_action_result,
            "total_reward": self.rewards.total,
        }
