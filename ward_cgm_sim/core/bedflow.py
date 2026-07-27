"""Bed occupancy, the ED/admissions queue and the discharge pipeline.

The discharge pipeline is where the workflow half of the research question
lives: a patient becomes clinically ready, somebody has to review that, then
somebody has to actually support the discharge. Each stage the agent neglects
adds delay, which backs up into the ED queue and eventually into unsafe
overcrowding.

ACADEMIC MODEL ONLY - simplified placeholders, not clinical guidance.
Stdlib only (ships to the browser build).
"""

import random

from .patient import (
    DischargeStage,
    EnrolmentStatus,
    Location,
    PatientState,
    sample_patient,
)


class WardFlow:
    """Owns the bed array, the admissions queue and discharge progression.

    ``rng`` is the ward-level stream: arrivals and the sampling of new
    patients. Everything that happens to an individual patient - transfers,
    discharge readiness - is drawn from that patient's own stream, so that an
    intervention on one patient cannot perturb another's trajectory.
    """

    def __init__(self, rng: random.Random, cfg, stream_seed: int = 0):
        self.cfg = cfg
        self.rng = rng
        self.stream_seed = stream_seed
        self.n_beds = cfg.ward.n_beds
        self.beds: list[PatientState | None] = [None] * self.n_beds
        self.queue: list[PatientState] = []
        self.next_patient_id = 0
        self.discharged: list[PatientState] = []

        # Cumulative counters used for KPIs and rewards.
        self.total_admissions = 0
        self.total_discharges = 0
        self.discharge_delay_steps = 0
        self.queue_length_history: list[int] = []

        self._populate_initial_ward()

    # ------------------------------------------------------------------
    def _new_patient(self, bed: int, step: int) -> PatientState:
        patient = sample_patient(
            self.rng, self.next_patient_id, bed, self.cfg, step, self.stream_seed
        )
        self.next_patient_id += 1
        return patient

    def _populate_initial_ward(self) -> None:
        wc = self.cfg.ward
        pc = self.cfg.patients
        n_occupied = int(round(self.n_beds * wc.initial_occupancy))
        beds = list(range(self.n_beds))
        self.rng.shuffle(beds)
        for bed in beds[:n_occupied]:
            patient = self._new_patient(bed, 0)
            # Existing patients are already part-way through their stay, spread
            # uniformly through it so that a realistic share are near discharge.
            expected_steps = patient.expected_los_hours * 60 / self.cfg.minutes_per_step
            patient.steps_on_ward = int(self.rng.uniform(0.0, expected_steps))
            self._maybe_enrol_at_handover(patient, pc)
            self.beds[bed] = patient
        for _ in range(self.rng.randint(*wc.initial_queue)):
            waiting = self._new_patient(-1, 0)
            waiting.location = Location.WALKING
            self.queue.append(waiting)

    def _maybe_enrol_at_handover(self, patient: PatientState, pc) -> None:
        """Mark, and in the telemetry arm enrol, the handover cohort.

        The agent inherits this cohort at handover. It knows *that* they are
        enrolled (that is on the dashboard) but not *why* they were judged
        eligible - the notes still have to be reviewed to confirm that they
        still qualify, which is what makes de-enrolment a real task.

        The cohort is selected identically whether or not telemetry is enabled,
        and the draws happen either way. That is what lets the counterfactual
        arm report outcomes for "the patients who would have been monitored",
        which is the only like-for-like comparison available.
        """
        from .eligibility import hard_exclusions  # local import: avoids a cycle

        if hard_exclusions(patient) or not patient.will_consent:
            return
        if self.rng.random() >= pc.initial_enrolled_fraction:
            return

        patient.telemetry_cohort = True
        if not self.cfg.telemetry_enabled:
            return

        patient.consent_asked = True
        patient.enrolment = EnrolmentStatus.ENROLLED
        patient.enrolled_step = 0
        patient.knowledge.consent_asked_step = 0
        patient.knowledge.known_consented = True
        # Seed some sensor history so rate-of-change alarms work from step 0.
        # The sensor bias itself is drawn from the sensor stream by the engine.
        patient.cgm_history.extend([patient.true_glucose] * 6)

    # ------------------------------------------------------------------
    @property
    def occupied_beds(self) -> int:
        return sum(1 for p in self.beds if p is not None)

    @property
    def free_beds(self) -> int:
        return self.n_beds - self.occupied_beds

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    def patients(self):
        for patient in self.beds:
            if patient is not None:
                yield patient

    def patient_at_bed(self, bed: int) -> PatientState | None:
        if 0 <= bed < self.n_beds:
            return self.beds[bed]
        return None

    def arrival_rate(self, step: int) -> float:
        wc = self.cfg.ward
        if wc.peak_start_step <= step <= wc.peak_end_step:
            return wc.arrival_rate_peak
        return wc.arrival_rate_base

    # ------------------------------------------------------------------
    def step(self, step: int) -> dict:
        """Advance arrivals, admissions and the discharge pipeline one step."""
        wc = self.cfg.ward
        events = {"admitted": 0, "discharged": 0, "arrived": 0, "became_ready": 0}

        # ED / admissions arrivals.
        rate = self.arrival_rate(step)
        while self.rng.random() < rate:
            waiting = self._new_patient(-1, step)
            waiting.location = Location.WALKING
            self.queue.append(waiting)
            events["arrived"] += 1
            rate -= 1.0  # allows >1 arrival per step at high intensity

        # Admissions into free beds, in queue order.
        while self.queue and self.free_beds > 0:
            bed = self.beds.index(None)
            patient = self.queue.pop(0)
            patient.bed = bed
            patient.admitted_step = step
            # Visibly walks from the ward entrance to the bed.
            patient.location = Location.WALKING
            # Patient's own stream: how many patients get admitted this step
            # depends on bed availability, which the agent influences.
            patient.walk_steps_left = patient.rng.randint(1, 3)
            patient.walk_total_steps = patient.walk_steps_left
            patient.walk_purpose = "admission"
            self.beds[bed] = patient
            self.total_admissions += 1
            events["admitted"] += 1

        # Per-patient progression.
        for patient in list(self.patients()):
            patient.steps_on_ward += 1
            self._step_movement(patient)
            self._step_discharge(patient, step, events)
            self._step_transfer(patient)

        self.queue_length_history.append(self.queue_length)
        return events

    # ------------------------------------------------------------------
    def _step_movement(self, patient: PatientState) -> None:
        if patient.location is Location.WALKING and patient.walk_steps_left > 0:
            patient.walk_steps_left -= 1
            if patient.walk_steps_left <= 0:
                if patient.walk_purpose == "discharge":
                    self._complete_discharge(patient)
                else:
                    patient.location = Location.BED
                    patient.walk_purpose = None

    def _step_transfer(self, patient: PatientState) -> None:
        wc = self.cfg.ward
        if patient.location is Location.OFF_WARD:
            patient.transfer_steps_left -= 1
            if patient.transfer_steps_left <= 0:
                patient.location = Location.WALKING
                patient.walk_steps_left = patient.rng.randint(1, 2)
                patient.walk_total_steps = patient.walk_steps_left
                patient.walk_purpose = "return"
        elif patient.location is Location.BED:
            # Drawn unconditionally on the patient's own stream: whether they
            # go to imaging is not caused by their discharge paperwork, and
            # gating the draw would let the agent's actions shift it.
            transfer_draw = patient.rng.random()
            transfer_length = patient.rng.randint(*wc.transfer_steps)
            if (
                patient.discharge_stage is DischargeStage.NOT_READY
                and transfer_draw < wc.transfer_prob
            ):
                patient.location = Location.OFF_WARD
                patient.transfer_steps_left = transfer_length

    def _step_discharge(self, patient: PatientState, step: int, events: dict) -> None:
        wc = self.cfg.ward
        stage = patient.discharge_stage

        # Every patient draws exactly once per step for discharge progression,
        # whatever stage they are in, so that an agent action which changes the
        # stage cannot change how much of the stream this patient consumes.
        stage_draw = patient.rng.random()

        if stage is DischargeStage.NOT_READY:
            # Readiness is a function of progress through the expected stay, so
            # a patient documented as staying 48 hours or more cannot be
            # discharged an hour later - which would otherwise contradict the
            # very criterion used to enrol them.
            expected_steps = max(1.0, patient.expected_los_hours * 60 / self.cfg.minutes_per_step)
            progress = patient.steps_on_ward / expected_steps
            if progress >= wc.discharge_ready_early_fraction:
                # Ramps from 0 at the earliest point to the full hazard at the
                # end of the expected stay and beyond.
                span = max(1e-6, 1.0 - wc.discharge_ready_early_fraction)
                scale = min(1.0, (progress - wc.discharge_ready_early_fraction) / span)
                if stage_draw < wc.discharge_ready_prob * scale:
                    patient.discharge_stage = DischargeStage.READY
                    patient.discharge_ready_step = step
                    events["became_ready"] += 1
            return

        if stage in (DischargeStage.READY, DischargeStage.REVIEWED):
            # Every step a ready patient sits in a bed is avoidable delay.
            self.discharge_delay_steps += 1
            # Background staff push it along on their own, slowly.
            if stage is DischargeStage.READY:
                if stage_draw < wc.background_review_prob:
                    patient.discharge_stage = DischargeStage.REVIEWED
            elif stage_draw < wc.background_support_prob:
                patient.discharge_stage = DischargeStage.SUPPORTED
                patient.discharge_prep_steps_left = wc.discharge_steps_after_support
            return

        if stage is DischargeStage.SUPPORTED:
            # Paperwork and logistics run down while the patient is still in
            # bed; when they finish, the patient walks off the ward and
            # _step_movement completes the discharge.
            if patient.walk_purpose == "discharge":
                return  # already walking out
            patient.discharge_prep_steps_left -= 1
            if patient.discharge_prep_steps_left <= 0:
                self._start_discharge_walk(patient)

    def _start_discharge_walk(self, patient: PatientState) -> None:
        patient.location = Location.WALKING
        patient.walk_purpose = "discharge"
        patient.walk_steps_left = 2
        patient.walk_total_steps = 2

    def support_discharge(self, patient: PatientState) -> bool:
        """Agent action: push a reviewed patient into the discharge pipeline."""
        if patient.discharge_stage is not DischargeStage.REVIEWED:
            return False
        patient.discharge_stage = DischargeStage.SUPPORTED
        patient.discharge_prep_steps_left = self.cfg.ward.discharge_steps_after_support
        return True

    def _complete_discharge(self, patient: PatientState) -> None:
        if 0 <= patient.bed < self.n_beds and self.beds[patient.bed] is patient:
            self.beds[patient.bed] = None
        patient.location = Location.DISCHARGED
        patient.discharge_stage = DischargeStage.DISCHARGED
        self.discharged.append(patient)
        self.total_discharges += 1

    # ------------------------------------------------------------------
    def prioritise_bedflow(self, step: int) -> int:
        """Agent action: chase the bed-flow backlog.

        Moves one ready patient to reviewed and speeds up the queue by pulling
        a waiting patient into any free bed. Returns how many patients moved.
        """
        moved = 0
        for patient in self.patients():
            if patient.discharge_stage is DischargeStage.READY:
                patient.discharge_stage = DischargeStage.REVIEWED
                patient.knowledge.discharge_reviewed_step = step
                patient.knowledge.known_discharge_ready = True
                moved += 1
                break
        while self.queue and self.free_beds > 0:
            bed = self.beds.index(None)
            waiting = self.queue.pop(0)
            waiting.bed = bed
            waiting.admitted_step = step
            waiting.location = Location.WALKING
            waiting.walk_steps_left = 1
            waiting.walk_total_steps = 1
            waiting.walk_purpose = "admission"
            self.beds[bed] = waiting
            self.total_admissions += 1
            moved += 1
        return moved

    # ------------------------------------------------------------------
    @property
    def overcrowded(self) -> bool:
        return self.queue_length > self.cfg.ward.overcrowding_penalty_start

    @property
    def unsafe_overcrowding(self) -> bool:
        return self.queue_length >= self.cfg.ward.unsafe_queue_length

    @property
    def safe_occupancy(self) -> bool:
        return self.queue_length <= self.cfg.ward.safe_queue_length and self.free_beds > 0
