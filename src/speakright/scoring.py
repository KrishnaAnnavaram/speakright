"""Deterministic scores. The same input always gives the same score. No LLM takes part.

English:  score = 100 x (0.8 x (1 - min(1, PER)) + 0.2 x mean confidence)   (with confidences)
          score = 100 x (1 - min(1, PER))                                      (without confidences)
Mandarin: score = 100 x (0.6 x (1 - min(1, syllable error rate)) + 0.4 x tone accuracy)
Tone accuracy counts only target syllables with tone 1 to 4 (the neutral tone 5 is not scored).
"""
from __future__ import annotations

from dataclasses import dataclass

BANDS = ((90.0, "excellent"), (75.0, "good"), (50.0, "fair"), (0.0, "needs practice"))


def band(score: float) -> str:
    for cut, name in BANDS:
        if score >= cut:
            return name
    return BANDS[-1][1]


@dataclass(frozen=True)
class Score:
    value: float
    band: str
    error_rate: float
    gop: float | None = None
    tone_accuracy: float | None = None

    def __post_init__(self):
        if not isinstance(self.value, float) or not 0.0 <= self.value <= 100.0:
            raise ValueError(f"score must be a float in [0, 100], got {self.value!r}")


def english_score(per: float, confidences: list[float] | None) -> Score:
    seg = 1.0 - min(1.0, per)
    if confidences:
        gop = sum(confidences) / len(confidences)
        value = 100.0 * (0.8 * seg + 0.2 * gop)
    else:
        gop = None
        value = 100.0 * seg
    value = round(float(value), 1)
    return Score(value, band(value), round(per, 4), None if gop is None else round(gop, 4))


def mandarin_score(syllable_error_rate: float, tone_accuracy: float | None) -> Score:
    seg = 1.0 - min(1.0, syllable_error_rate)
    value = 100.0 * (seg if tone_accuracy is None else 0.6 * seg + 0.4 * tone_accuracy)
    value = round(float(value), 1)
    return Score(value, band(value), round(syllable_error_rate, 4), None,
                 None if tone_accuracy is None else round(tone_accuracy, 4))
