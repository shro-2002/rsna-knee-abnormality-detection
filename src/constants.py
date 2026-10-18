"""Single source of truth for competition schema."""

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

ID_COL = "StudyInstanceUID"
SUBMISSION_COLUMNS = [ID_COL] + TARGETS
N_TARGETS = len(TARGETS)
