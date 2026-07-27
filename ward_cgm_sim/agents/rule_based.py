"""A hand-written nurse policy: the sensible baseline to beat.

The priority order encodes how the workflow is *supposed* to run:

  1. urgent alarm  -> walk to the bed, confirm with point-of-care, then treat
                      (or escalate if it is severe or recurrent)
  2. non-urgent alarm -> same, at lower priority
  3. silent sensor loss -> troubleshoot before the gap hides a real event
  4. enrolled patient who no longer qualifies -> de-enrol
  5. enrolment work on unreviewed patients -> notes, consent, enrol
  6. discharge and bed flow -> keep beds moving
  7. otherwise -> stand at the nurse station watching the dashboard

It reads the engine's *observable* surface only (it may inspect what the agent
has recorded in ``PatientKnowledge``, never a hidden field), so it is a fair
comparator for a learned policy rather than an oracle.

Stdlib only - this ships to the browser build.
"""

from ..core.actions import Action
from ..core.alarms import URGENT_ALARMS
from ..core.patient import DischargeStage, EnrolmentStatus, Location, Specialty


class RuleBasedAgent:
    """Greedy priority-driven nurse policy."""

    name = "rule_based"

    # How stale the agent's picture of the dashboard may get before it looks
    # again. Five steps is 25 minutes - roughly how often a nurse glances at a
    # central monitor or a handheld between other tasks.
    DASHBOARD_STALENESS = 5
    # How often an enrolled patient's continued eligibility is re-checked.
    ELIGIBILITY_REVIEW_INTERVAL = 48  # steps (4 hours)

    def __init__(self, seed: int | None = None):
        self.target_bed: int | None = None

    def reset(self) -> None:
        self.target_bed = None

    # ------------------------------------------------------------------
    def act(self, engine) -> int:
        goal = self._choose_goal(engine)

        # Keep the picture of the telemetry board fresh. Without this the agent
        # walks off doing paperwork and never learns that anybody is alarming -
        # a real failure mode, but not one a baseline should exhibit. It is
        # deliberately NOT allowed to interrupt an alarm response: you do not
        # walk away from a patient you are assessing to look at a screen.
        if goal is None or goal[1] != "alarm":
            if self._dashboard_is_stale(engine):
                return Action.CHECK_DASHBOARD

        if goal is None:
            return self._go_to_station(engine)

        bed, intent = goal
        here = engine.adjacent_bed()
        if here == bed:
            return self._act_on_patient(engine, bed, intent)
        return self._move_toward_bed(engine, bed)

    def _dashboard_is_stale(self, engine) -> bool:
        if not engine.cfg.telemetry_enabled:
            return False
        if engine.ward_map.at_station(engine.agent_x, engine.agent_y):
            return False  # standing at the board, it refreshes for free
        seen = engine.dashboard_seen_step
        if seen is None:
            return True
        return engine.step_index - seen > self.DASHBOARD_STALENESS

    # ------------------------------------------------------------------
    def _choose_goal(self, engine) -> tuple[int, str] | None:
        # An alarm can only be acted on if the patient is actually in the bed;
        # chasing somebody who is in theatre just burns the shift.
        alarms = [
            a
            for a in engine.visible_alarms()
            if (p := engine.flow.patient_at_bed(a.bed)) is not None and p.visible_at_bed
        ]

        urgent = [a for a in alarms if a.kind in URGENT_ALARMS]
        if urgent:
            urgent.sort(key=lambda a: (a.acknowledged_step is not None, -a.age(engine.step_index)))
            return urgent[0].bed, "alarm"

        if alarms:
            alarms.sort(key=lambda a: -a.age(engine.step_index))
            return alarms[0].bed, "alarm"

        # Silent sensor problems: no alarm ever fires for these.
        grace = engine.cfg.alarms.signal_loss_grace_steps
        for patient in engine.flow.patients():
            if patient.is_enrolled and patient.steps_since_valid_cgm > grace:
                return patient.bed, "sensor"

        # Bed pressure outranks enrolment paperwork: a queue that reaches the
        # unsafe threshold ends the shift, so it cannot wait behind admin.
        if engine.flow.queue_length > engine.cfg.ward.safe_queue_length:
            for patient in engine.flow.patients():
                if patient.discharge_stage in (DischargeStage.READY, DischargeStage.REVIEWED):
                    return patient.bed, "discharge"

        # De-enrol anyone the agent has already reviewed and found ineligible.
        for patient in engine.flow.patients():
            if not patient.is_enrolled:
                continue
            k = patient.knowledge
            if not k.knows_full_eligibility():
                continue
            still_eligible = (
                bool(k.known_has_diabetes)
                and bool(k.known_two_or_more_injections)
                and bool(k.known_expected_los_at_least_48h)
                and bool(k.known_has_capacity)
                and not bool(k.known_excluded)
            )
            if not still_eligible:
                return patient.bed, "deenrol"

        # Enrolment pipeline for patients not yet on telemetry.
        for patient in engine.flow.patients():
            if patient.enrolment in (EnrolmentStatus.ENROLLED, EnrolmentStatus.DECLINED):
                continue
            if patient.enrolment is EnrolmentStatus.DEENROLLED:
                continue
            k = patient.knowledge
            if not k.knows_full_eligibility():
                return patient.bed, "review_notes"
            looks_eligible = (
                bool(k.known_has_diabetes)
                and bool(k.known_two_or_more_injections)
                and bool(k.known_expected_los_at_least_48h)
                and bool(k.known_has_capacity)
                and not bool(k.known_excluded)
            )
            if not looks_eligible:
                continue
            if k.consent_asked_step is None:
                return patient.bed, "consent"
            if k.known_consented:
                return patient.bed, "enrol"

        # Periodic re-review of enrolled patients, to catch a regimen change or
        # a transition to end-of-life care that makes somebody ineligible.
        for patient in engine.flow.patients():
            if not patient.is_enrolled:
                continue
            last = patient.knowledge.eligibility_reviewed_step
            if last is None or engine.step_index - last > self.ELIGIBILITY_REVIEW_INTERVAL:
                return patient.bed, "review_eligibility"

        # Discharge work.
        for patient in engine.flow.patients():
            if patient.discharge_stage in (DischargeStage.READY, DischargeStage.REVIEWED):
                return patient.bed, "discharge"

        return None

    # ------------------------------------------------------------------
    def _act_on_patient(self, engine, bed: int, intent: str) -> int:
        patient = engine.flow.patient_at_bed(bed)
        if patient is None:
            return Action.WAIT

        if intent == "alarm":
            alarm = engine.active_alarms.get(bed)
            if alarm is None:
                return Action.CHECK_PATIENT
            if alarm.acknowledged_step is None:
                return Action.RESPOND_ALARM
            # Confirm with point-of-care before treating anything significant.
            if alarm.needs_poc and alarm.poc_confirmed_step is None:
                return Action.POC_GLUCOSE_TEST

            poc = patient.knowledge.last_poc_value
            ac = engine.cfg.alarms
            if poc is None:
                return Action.POC_GLUCOSE_TEST
            if poc < ac.severe_hypo_threshold:
                # Severe: escalate as well as treat.
                if engine._is_recurrent(patient):
                    return Action.ESCALATE
                return Action.TREAT_HYPO
            if poc < ac.hypo_threshold:
                return Action.TREAT_HYPO
            if poc > patient.hyper_threshold(ac.hyper_threshold_default):
                return Action.TREAT_HYPER
            # Point-of-care disagrees with the alarm: trust point-of-care and
            # do not treat. Checking the patient clears the visit.
            return Action.CHECK_PATIENT

        if intent == "sensor":
            return Action.TROUBLESHOOT_SENSOR
        if intent == "deenrol":
            return Action.DEENROL
        if intent == "review_eligibility":
            return Action.REVIEW_ELIGIBILITY
        if intent == "review_notes":
            return Action.REVIEW_NOTES
        if intent == "consent":
            if patient.location is not Location.BED:
                return Action.WAIT
            return Action.ASK_CONSENT
        if intent == "enrol":
            return Action.ENROL
        if intent == "discharge":
            # Ask the right colleague when one can take the decision, otherwise
            # do it yourself.
            if patient.discharge_stage is DischargeStage.READY:
                if engine.staff.is_available("surgeon") and patient.specialty is Specialty.SURGICAL:
                    return Action.ASK_HELP_SURGEON
                if engine.staff.is_available("doctor") and patient.specialty is Specialty.MEDICAL:
                    return Action.ASK_HELP_DOCTOR
            return Action.SUPPORT_DISCHARGE

        return Action.WAIT

    # ------------------------------------------------------------------
    def _move_toward_bed(self, engine, bed: int) -> int:
        approach = engine.ward_map.approach_tile(bed)
        if approach is None:
            return Action.WAIT
        return self._step_toward(engine, approach)

    def _go_to_station(self, engine) -> int:
        if engine.ward_map.at_station(engine.agent_x, engine.agent_y):
            # Standing at the board: keep the queue moving, else watch it.
            if engine.flow.queue_length > engine.cfg.ward.safe_queue_length:
                return Action.PRIORITISE_BEDFLOW
            return Action.CHECK_DASHBOARD
        target = engine.ward_map.station_tiles[0]
        # Stand beside the station rather than on it.
        beside = (target[0] - 1, target[1])
        return self._step_toward(engine, beside)

    def _step_toward(self, engine, goal: tuple[int, int]) -> int:
        start = (engine.agent_x, engine.agent_y)
        if start == goal:
            return Action.WAIT
        nxt = engine.ward_map.next_step_toward(start, goal)
        if nxt is None:
            return Action.WAIT
        dx, dy = nxt[0] - start[0], nxt[1] - start[1]
        if dx == 1:
            return Action.MOVE_RIGHT
        if dx == -1:
            return Action.MOVE_LEFT
        if dy == 1:
            return Action.MOVE_DOWN
        if dy == -1:
            return Action.MOVE_UP
        return Action.WAIT
