"""Needleman-Wunsch alignment of a canonical sequence and a recognised sequence.

Each operation is `match`, `sub`, `del` (canonical unit missing) or `ins` (extra recognised unit).
One insertion shifts nothing after it. PER = (S + D + I) / N, with N = canonical length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

from .phones import VOWELS, phone_class
from .pinyin import Syllable

T = TypeVar("T")


@dataclass(frozen=True)
class Op(Generic[T]):
    kind: str            # match / sub / del / ins
    ref: T | None
    hyp: T | None
    ref_index: int | None
    hyp_index: int | None


def phone_sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if (a in VOWELS) != (b in VOWELS):
        return 1.6  # a vowel for a consonant is never a cheap substitution
    return 0.8 if phone_class(a) == phone_class(b) else 1.0


def syllable_sub_cost(a: Syllable, b: Syllable) -> float:
    return 0.4 * (a.initial != b.initial) + 0.4 * (a.final != b.final) + 0.2 * (a.tone != b.tone)


def align(ref: Sequence[T], hyp: Sequence[T], sub_cost: Callable[[T, T], float], gap: float = 1.0) -> list[Op[T]]:
    n, m = len(ref), len(hyp)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]), D[i - 1][j] + gap, D[i][j - 1] + gap)
    ops: list[Op[T]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]))) < 1e-9:
            kind = "match" if sub_cost(ref[i - 1], hyp[j - 1]) == 0 else "sub"
            ops.append(Op(kind, ref[i - 1], hyp[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + gap)) < 1e-9:
            ops.append(Op("del", ref[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append(Op("ins", None, hyp[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def error_rate(ops: Sequence[Op], n_ref: int) -> float:
    errors = sum(op.kind != "match" for op in ops)
    return errors / n_ref if n_ref else (1.0 if errors else 0.0)
