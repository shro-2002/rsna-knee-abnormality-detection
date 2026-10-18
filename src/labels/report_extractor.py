"""Multilingual weak-label extractor: knee-MRI report -> 12 soft targets + abstain mask.

Approach (NegEx-lite, language-agnostic):
  - accent-fold + lowercase; split into clauses on sentence/line boundaries;
  - a target fires on a clause containing its pathology term (and anatomy term, if required);
  - each pathology occurrence is negated if a negation cue precedes it within a char window;
  - uncertainty cues in the clause downgrade confidence;
  - unmentioned targets are weakly-negative (if routinely assessed) or ABSTAINED (mask=0).

Outputs soft labels in [0,1] and a mask (1=use in weak supervision, 0=abstain). These are weak
labels for TRAINING only; they are never treated as ground truth. Quality is measured against the
58 gold studies (see scripts/eval_extractor.py), not assumed.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from src.constants import TARGETS
from src.labels.lexicon import (ANATOMY, NEGATION, PATH, TARGET_RULES, UNCERTAINTY)

_NEG_WINDOW = 30  # chars before a pathology hit scanned for a negation cue
_CLAUSE_SPLIT = re.compile(r"[.;:\n\r]|(?<=\))\s|•|·|-\s")

# soft-label policy
_SOFT = {"positive": 0.92, "uncertain": 0.68, "negated": 0.06,
         "unmentioned_assessed": 0.22, "abstain": 0.0}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c.strip()]


def _any_in(terms: list[str], clause: str) -> bool:
    return any(t in clause for t in terms)


def _negated_at(clause: str, pos: int) -> bool:
    window = clause[max(0, pos - _NEG_WINDOW):pos]
    return any(cue in window for cue in NEGATION)


def _classify_target(clauses: list[str], rule) -> str:
    kind, anat_key, path_key, assessed = rule
    path_terms = PATH[path_key]
    anat_terms = ANATOMY[anat_key] if anat_key else None
    mentioned = False
    saw_positive = saw_uncertain = saw_negated = False
    for clause in clauses:
        # for anatomy x pathology targets, the anatomy must be named in the same clause
        if anat_terms is not None and not _any_in(anat_terms, clause):
            continue
        # find pathology hits
        hits = [m.start() for t in path_terms for m in re.finditer(re.escape(t), clause)]
        if not hits:
            continue
        mentioned = True
        uncertain = _any_in(UNCERTAINTY, clause)
        for pos in hits:
            if _negated_at(clause, pos):
                saw_negated = True
            elif uncertain:
                saw_uncertain = True
            else:
                saw_positive = True
    if saw_positive:
        return "positive"
    if saw_uncertain:
        return "uncertain"
    if saw_negated or mentioned:
        return "negated"
    return "unmentioned_assessed" if assessed else "abstain"


def extract_one(report: str) -> dict:
    clauses = _clauses(fold(report))
    out = {}
    for t in TARGETS:
        status = _classify_target(clauses, TARGET_RULES[t])
        out[t] = {"status": status, "soft": _SOFT[status],
                  "mask": 0 if status == "abstain" else 1}
    return out


def extract_dataframe(df: pd.DataFrame, id_col: str = "StudyInstanceUID",
                      report_col: str = "Report"):
    rows_soft, rows_mask, rows_status = [], [], []
    for _, r in df.iterrows():
        res = extract_one(r[report_col])
        rows_soft.append({id_col: r[id_col], **{t: res[t]["soft"] for t in TARGETS}})
        rows_mask.append({id_col: r[id_col], **{t: res[t]["mask"] for t in TARGETS}})
        rows_status.append({id_col: r[id_col], **{t: res[t]["status"] for t in TARGETS}})
    return (pd.DataFrame(rows_soft), pd.DataFrame(rows_mask), pd.DataFrame(rows_status))


if __name__ == "__main__":
    samples = [
        "Rotura de menisco interno. Artrosis femorotibial medial. Derrame.",
        "No se observa rotura del ligamento cruzado anterior. Menisco interno sin alteraciones.",
        "Geen scheur van de voorste kruisband. Mediale meniscus intact.",
        "Possible tear of the lateral meniscus. Joint effusion present.",
    ]
    for s in samples:
        res = extract_one(s)
        hot = {t: res[t]["status"] for t in TARGETS if res[t]["status"] != "abstain"
               and res[t]["soft"] >= 0.5}
        print(f"\n{s}\n -> fired: {hot}")
