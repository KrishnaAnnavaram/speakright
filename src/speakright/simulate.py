"""Synthetic learners, a noisy phone recogniser and the old word-level method, for the evaluation.

These simulations have known, labelled errors. They prove that the pipeline finds errors that
are in the phones. They are not a measure on real speech.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .g2p_en import BUILTIN_LEXICON, WordPhones
from .phones import CLASSES, VOWELS, phone_class

EN_RULES: dict[str, list[tuple[str, str]]] = {
    "TH": [("T", "th-stopping"), ("S", "th-fronting"), ("F", "th-fronting")],
    "DH": [("D", "th-stopping"), ("Z", "th-fronting")],
    "R": [("L", "r-l-confusion")], "L": [("R", "r-l-confusion")],
    "V": [("W", "v-w-confusion"), ("B", "v-b-confusion")], "W": [("V", "v-w-confusion")],
    "IY": [("IH", "vowel-length")], "IH": [("IY", "vowel-length")],
    "SH": [("S", "sh-s-confusion")], "Z": [("S", "final-devoicing")],
}
STOPS = CLASSES["stop"]


@dataclass
class SimUtterance:
    words: list[WordPhones]
    actual: list[str]
    error_ref_indices: set[int]       # canonical positions with a real error (ins -> previous position)
    wrong_words: set[int]
    n_errors: int


def simulate_english(words: list[WordPhones], rng: np.random.Generator, p_err: float = 0.3) -> SimUtterance:
    actual: list[str] = []
    errs: set[int] = set()
    wrong: set[int] = set()
    idx = 0
    n = 0
    for wi, wp in enumerate(words):
        for k, p in enumerate(wp.phones):
            last = k == len(wp.phones) - 1
            r = rng.random()
            if p in EN_RULES and r < p_err:
                rep, _ = EN_RULES[p][int(rng.integers(len(EN_RULES[p])))]
                actual.append(rep)
                errs.add(idx); wrong.add(wi); n += 1
            elif last and p not in VOWELS and len(wp.phones) > 2 and r < p_err * 0.3:
                errs.add(idx); wrong.add(wi); n += 1          # final consonant deletion
            elif last and p in STOPS and r < p_err * 0.5:
                actual.extend([p, "AH"])                      # vowel epenthesis
                errs.add(idx); wrong.add(wi); n += 1
            else:
                actual.append(p)
            idx += 1
    return SimUtterance(words, actual, errs, wrong, n)


def noisy_recognizer(actual: list[str], rng: np.random.Generator, noise: float = 0.05) -> tuple[list[str], list[float]]:
    """Recognition noise: a same-class substitution or a deletion at rate `noise`, with confidences."""
    out, conf = [], []
    for p in actual:
        r = rng.random()
        if r < noise * 0.25:
            continue
        if r < noise:
            same = sorted(CLASSES[phone_class(p)] - {p})
            out.append(same[int(rng.integers(len(same)))])
            conf.append(float(rng.uniform(0.3, 0.7)))
        else:
            out.append(p)
            conf.append(float(rng.uniform(0.75, 1.0)))
    return out, conf


def legacy_word_flags(words: list[WordPhones], wrong: set[int], rng: np.random.Generator,
                      p_lm_fix: float = 0.85) -> set[int]:
    """The old method: a word ASR (its language model returns the target word for most mispronounced
    words), then a position-wise comparison of dictionary pronunciations. Returns flagged word indices."""
    vocab = sorted(BUILTIN_LEXICON)
    recognised = []
    for wi, wp in enumerate(words):
        if wi in wrong and rng.random() > p_lm_fix:
            recognised.append(vocab[int(rng.integers(len(vocab)))])
        else:
            recognised.append(wp.word)
    return {i for i, (wp, w) in enumerate(zip(words, recognised)) if wp.word != w}


ZH_TONE_ERRORS = {1: [4, 2], 2: [3, 1], 3: [2, 4], 4: [1, 2]}
ZH_INITIAL_ERRORS = {"zh": "z", "ch": "c", "sh": "s", "n": "l", "l": "n"}
ZH_FINAL_ERRORS = {"ang": "an", "an": "ang", "eng": "en", "en": "eng", "ing": "in", "in": "ing"}


def simulate_mandarin(ref: list, rng: np.random.Generator, p_tone: float = 0.3, p_seg: float = 0.15):
    """Return spoken syllables (Syllable list), tone-error positions and segment-error positions."""
    from .pinyin import Syllable

    spoken, tone_err, seg_err = [], set(), set()
    for i, s in enumerate(ref):
        ini, fin, tone = s.initial, s.final, s.tone
        if tone in ZH_TONE_ERRORS and rng.random() < p_tone:
            opts = ZH_TONE_ERRORS[tone]
            tone = opts[int(rng.integers(len(opts)))]
            tone_err.add(i)
        if rng.random() < p_seg:
            if ini in ZH_INITIAL_ERRORS:
                ini = ZH_INITIAL_ERRORS[ini]
                seg_err.add(i)
            elif fin in ZH_FINAL_ERRORS:
                fin = ZH_FINAL_ERRORS[fin]
                seg_err.add(i)
        spoken.append(Syllable(ini, fin, tone))
    return spoken, tone_err, seg_err
