"""Background staff availability and the escalation model.

Every role is a hidden two-state Markov chain (available / busy). The agent
sees only a coarse, deliberately lossy summary on the observation - "roughly
how many people are about" - and has to actually ask to find out whether a
specific role can help right now.

Asking the wrong role wastes the step: an HCA cannot prescribe, a surgeon has
no interest in a medical patient's insulin. That is what makes the four
role-specific request actions a learning problem rather than a formality.

ACADEMIC MODEL ONLY - simplified placeholders, not clinical guidance.
Stdlib only (ships to the browser build).
"""

import random

# Which role is the right one to ask for a given kind of help.
ROLE_COMPETENCIES: dict[str, frozenset[str]] = {
    "hca": frozenset({"check", "poc", "discharge_prep", "bedflow"}),
    "nurse": frozenset({"check", "poc", "treat", "discharge_prep", "bedflow"}),
    "doctor": frozenset({"review", "prescribe", "medical_escalation", "discharge_decision"}),
    "surgeon": frozenset({"review", "surgical_discharge_decision"}),
    "diabetes": frozenset({"medical_escalation", "review", "prescribe"}),
}

# Roles that can accept a severe/recurrent glycaemic escalation.
ESCALATION_ROLES = ("diabetes", "doctor")


class StaffPool:
    """Hidden availability state for the ward's background staff."""

    def __init__(self, rng: random.Random, cfg):
        self.cfg = cfg
        self.rng = rng
        sc = cfg.staff
        self.available: dict[str, bool] = {
            role: rng.random() < sc.availability_start[role] for role in sc.roles
        }
        self.workload: float = 0.0
        self.requests_made: int = 0
        self.wrong_role_requests: int = 0

    # ------------------------------------------------------------------
    def step(self) -> None:
        sc = self.cfg.staff
        for role in sc.roles:
            if self.available[role]:
                if self.rng.random() < sc.become_busy_prob:
                    self.available[role] = False
            else:
                if self.rng.random() < sc.become_available_prob:
                    self.available[role] = True
        self.workload = max(0.0, self.workload - sc.workload_decay)

    # ------------------------------------------------------------------
    def coarse_availability(self) -> int:
        """The lossy summary the agent can see without asking.

        0 = skeleton crew, 1 = stretched, 2 = comfortable. Deliberately coarse:
        it never tells the agent whether the *specific* role they need is free.
        """
        n_available = sum(1 for role in self.cfg.staff.roles if self.available[role])
        if n_available <= 1:
            return 0
        if n_available <= 3:
            return 1
        return 2

    def is_available(self, role: str) -> bool:
        return self.available.get(role, False)

    def request(self, role: str, task: str) -> tuple[bool, str]:
        """Ask a role for help with a task.

        Returns ``(succeeded, outcome)`` where outcome is one of
        ``"helped"``, ``"busy"``, ``"wrong_role"``.
        """
        sc = self.cfg.staff
        self.requests_made += 1
        self.workload += sc.workload_per_request

        if task not in ROLE_COMPETENCIES.get(role, frozenset()):
            self.wrong_role_requests += 1
            return False, "wrong_role"
        if not self.available.get(role, False):
            return False, "busy"
        # Helping occupies them for a while.
        self.available[role] = False
        return True, "helped"

    def escalate(self) -> tuple[bool, str]:
        """Escalate a severe or recurrent glycaemic event.

        Routed to the diabetes team first, then the on-call doctor.
        """
        sc = self.cfg.staff
        self.requests_made += 1
        self.workload += sc.workload_per_request
        for role in ESCALATION_ROLES:
            if self.available.get(role, False):
                self.available[role] = False
                return True, role
        return False, "busy"

    @property
    def overloaded(self) -> bool:
        return self.workload > self.cfg.staff.workload_penalty_threshold
