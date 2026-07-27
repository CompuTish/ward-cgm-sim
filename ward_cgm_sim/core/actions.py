"""Action space definition.

24 discrete actions: 4 movement + 20 interactions. Interactions are targeted at
the bed the agent is standing next to, which keeps the action space flat while
making movement meaningful (a POMDP with a factored per-patient action space
would be far harder to train and no more faithful).

Note two spec items that are intentionally not separate actions:

* "avoid enrolling an ineligible patient" is expressed by *not* selecting
  ENROL - choosing it wrongly is penalised, so avoidance is learned behaviour
  rather than a button.
* "ask HCA/nurse/doctor/surgeon for help" IS split into four actions, so that
  choosing the right colleague is something the agent learns.

Stdlib only (ships to the browser build).
"""

from enum import IntEnum


class Action(IntEnum):
    # --- movement -----------------------------------------------------
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3

    # --- information gathering ----------------------------------------
    CHECK_DASHBOARD = 4
    CHECK_PATIENT = 5
    REVIEW_NOTES = 6

    # --- enrolment workflow -------------------------------------------
    ASK_CONSENT = 7
    ENROL = 8
    REVIEW_ELIGIBILITY = 9
    DEENROL = 10

    # --- clinical response --------------------------------------------
    RESPOND_ALARM = 11
    POC_GLUCOSE_TEST = 12
    TREAT_HYPO = 13
    TREAT_HYPER = 14
    ESCALATE = 15

    # --- asking colleagues (role-specific on purpose) -----------------
    ASK_HELP_HCA = 16
    ASK_HELP_NURSE = 17
    ASK_HELP_DOCTOR = 18
    ASK_HELP_SURGEON = 19

    # --- technical and workflow ---------------------------------------
    TROUBLESHOOT_SENSOR = 20
    SUPPORT_DISCHARGE = 21
    PRIORITISE_BEDFLOW = 22
    WAIT = 23


N_ACTIONS = len(Action)

MOVEMENT_ACTIONS = frozenset(
    {Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT}
)

MOVEMENT_DELTAS = {
    Action.MOVE_UP: (0, -1),
    Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0),
    Action.MOVE_RIGHT: (1, 0),
}

# Actions that need a patient in an adjacent bed.
PATIENT_TARGETED_ACTIONS = frozenset(
    {
        Action.CHECK_PATIENT,
        Action.REVIEW_NOTES,
        Action.ASK_CONSENT,
        Action.ENROL,
        Action.REVIEW_ELIGIBILITY,
        Action.DEENROL,
        Action.RESPOND_ALARM,
        Action.POC_GLUCOSE_TEST,
        Action.TREAT_HYPO,
        Action.TREAT_HYPER,
        Action.ESCALATE,
        Action.TROUBLESHOOT_SENSOR,
        Action.SUPPORT_DISCHARGE,
    }
)

# Administrative / non-urgent actions. Choosing one of these while an urgent
# alarm is live counts as unsafe prioritisation.
ADMIN_ACTIONS = frozenset(
    {
        Action.REVIEW_NOTES,
        Action.ASK_CONSENT,
        Action.ENROL,
        Action.REVIEW_ELIGIBILITY,
        Action.DEENROL,
        Action.SUPPORT_DISCHARGE,
        Action.PRIORITISE_BEDFLOW,
        Action.WAIT,
    }
)

ROLE_ACTIONS = {
    Action.ASK_HELP_HCA: "hca",
    Action.ASK_HELP_NURSE: "nurse",
    Action.ASK_HELP_DOCTOR: "doctor",
    Action.ASK_HELP_SURGEON: "surgeon",
}

ACTION_LABELS = {
    Action.MOVE_UP: "Move up",
    Action.MOVE_DOWN: "Move down",
    Action.MOVE_LEFT: "Move left",
    Action.MOVE_RIGHT: "Move right",
    Action.CHECK_DASHBOARD: "Check telemetry dashboard",
    Action.CHECK_PATIENT: "Check patient",
    Action.REVIEW_NOTES: "Review notes / drug chart",
    Action.ASK_CONSENT: "Ask for verbal consent",
    Action.ENROL: "Enrol patient",
    Action.REVIEW_ELIGIBILITY: "Review enrolled patient eligibility",
    Action.DEENROL: "De-enrol patient",
    Action.RESPOND_ALARM: "Respond to alarm",
    Action.POC_GLUCOSE_TEST: "Point-of-care capillary glucose",
    Action.TREAT_HYPO: "Treat hypoglycaemia (placeholder pathway)",
    Action.TREAT_HYPER: "Treat hyperglycaemia (placeholder pathway)",
    Action.ESCALATE: "Escalate to medical / diabetes team",
    Action.ASK_HELP_HCA: "Ask HCA for help",
    Action.ASK_HELP_NURSE: "Ask nurse for help",
    Action.ASK_HELP_DOCTOR: "Ask doctor for help",
    Action.ASK_HELP_SURGEON: "Ask surgeon for help",
    Action.TROUBLESHOOT_SENSOR: "Troubleshoot sensor / signal loss",
    Action.SUPPORT_DISCHARGE: "Support discharge preparation",
    Action.PRIORITISE_BEDFLOW: "Prioritise bed-flow tasks",
    Action.WAIT: "Wait",
}
