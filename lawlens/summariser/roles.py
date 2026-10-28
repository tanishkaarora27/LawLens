"""Rhetorical roles of a judgment, with the zero-shot cues used to find them.

No model is trained here. Three weak signals are combined (see scoring.py):
  1. regex cue phrases typical of each part of an Indian judgment,
  2. similarity to a few prototype sentences, using a pretrained sentence encoder,
  3. where in the document the paragraph sits (facts early, decision at the end).
"""

from __future__ import annotations

import re

import numpy as np

ROLES = ["facts", "issues", "arguments", "reasoning", "decision"]

ROLE_TITLES = {
    "facts": "Facts",
    "issues": "Issues",
    "arguments": "Arguments",
    "reasoning": "Court's reasoning",
    "decision": "Decision / outcome",
}

_RAW_CUES = {
    "facts": [
        r"\b(filed|lodged|registered)\b",
        r"\bFIR\b|\bfirst information report\b|\bcharge[- ]?sheet\b",
        r"\bbrief facts\b|\bfacts of the case\b|\bfacts leading\b|\bcase of the prosecution\b|\bprosecution case\b|\bbackground\b",
        r"\b(was|were) (appointed|arrested|convicted|acquitted|terminated|dismissed|detained|registered|posted|promoted)\b",
        r"\bon \d{1,2}(st|nd|rd|th)?[\s./-]+\w+[\s./-]+\d{2,4}\b|\bdated\b",
        r"\b(trial court|high court|sessions (court|judge)|tribunal|magistrate) (convicted|acquitted|held|allowed|dismissed|passed|decreed)\b",
        r"\bthe appellant\b.*\bthe respondent\b",
    ],
    "issues": [
        r"\bwhether\b",
        r"\bquestion(s)? (of law )?(that |which )?(arise|arises|falls?|fall|for consideration|for determination)\b",
        r"\bpoints? (for|of) determination\b|\bissues? (framed|for consideration|for determination|that arise)\b",
        r"\b(falls?|arises?) for (our )?consideration\b",
        r"\bsubstantial question of law\b",
    ],
    "arguments": [
        r"\blearned (senior )?(counsel|advocate)\b|\blearned (additional|solicitor|attorney)\b",
        r"\b(submitted|submits|contended|contends|argued|argues|urged|urges|pleaded)\b",
        r"\bon behalf of the (appellant|respondent|petitioner|state)\b|\bappearing for\b",
        r"\b(Mr|Ms|Mrs|Shri|Smt|Dr)\. ?\w+",
        r"\bit was (further )?(submitted|contended|argued|urged)\b",
    ],
    "reasoning": [
        r"\bwe are of (the )?(view|opinion)\b|\bin our (view|opinion)\b|\bwe (find|hold|note|observe)\b",
        r"\b(therefore|hence|thus|consequently|accordingly)\b",
        r"\bin view of\b|\bhaving regard to\b|\bkeeping in view\b",
        r"\b(this court|apex court|supreme court|high court) (in|has|had|held|observed)\b|\bAIR \d{4}\b|\b\(\d{4}\) \d+ SCC\b|\bSCC\b",
        r"\b(principle|ratio|settled (law|position)|well[- ]settled|ingredients?)\b",
        r"\b(erred|perverse|unsustainable|untenable|not sustainable|cannot be sustained)\b",
    ],
    "decision": [
        r"\b(appeals?|petitions?|writ petition|special leave petition|slp|application)s? (is|are|stands?|stand) (allowed|dismissed|disposed|partly allowed|accordingly)\b",
        r"\b(allowed|dismissed|disposed of)\b",
        r"\b(set aside|quashed|upheld|confirmed|restored|remanded|remitted)\b",
        r"\bno order as to costs\b|\bparties (shall|will) bear their own costs\b|\bcosts? (are|is|of)\b",
        r"\bfor the (foregoing|aforesaid|above) reasons\b|\bin the result\b|\baccordingly\b",
    ],
}

CUES = {role: [re.compile(p, re.I) for p in pats] for role, pats in _RAW_CUES.items()}

PROTOTYPES = {
    "facts": [
        "The appellant filed a complaint and the police registered an FIR on the date of the incident.",
        "The brief facts of the case are that the respondent was employed and his services were terminated.",
        "The trial court convicted the accused and the High Court dismissed his appeal.",
    ],
    "issues": [
        "The question that arises for consideration is whether the High Court was justified in setting aside the order.",
        "The following issues fall for determination by this Court.",
        "Whether the impugned provision violates Article 14 of the Constitution.",
    ],
    "arguments": [
        "Learned counsel for the appellant submitted that the findings are perverse.",
        "It was contended on behalf of the respondent that the appeal is not maintainable.",
        "The Senior Advocate argued that the statute does not apply to the petitioner.",
    ],
    "reasoning": [
        "In our opinion, the High Court erred in law, since the principle laid down by this Court applies.",
        "We have considered the evidence and the precedents, and find that the ingredients of the offence are not made out.",
        "Therefore, in view of the settled position of law, the contention cannot be accepted.",
    ],
    "decision": [
        "The appeal is allowed and the judgment of the High Court is set aside.",
        "For the reasons stated above, the appeal is dismissed with no order as to costs.",
        "The conviction and sentence are upheld and the appeal stands disposed of.",
    ],
}

# Where in the document (0 = start, 1 = end) each role usually sits, and how loosely.
POS_CENTER = {"facts": 0.12, "issues": 0.30, "arguments": 0.38, "reasoning": 0.70, "decision": 0.96}
POS_SIGMA = {"facts": 0.15, "issues": 0.20, "arguments": 0.22, "reasoning": 0.25, "decision": 0.06}


def cue_score(text: str, role: str) -> float:
    """Share of the role's cue patterns that fire, saturating at two hits."""
    hits = sum(1 for p in CUES[role] if p.search(text))
    return min(1.0, hits / 2.0)


def position_prior(role: str, rel_pos: float) -> float:
    c, s = POS_CENTER[role], POS_SIGMA[role]
    return float(np.exp(-0.5 * ((rel_pos - c) / s) ** 2))
