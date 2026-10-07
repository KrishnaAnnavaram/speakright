"""Mandarin tones from the audio itself: F0 track, syllable segments, rule-based tone classes.

1. `f0_track`: normalised autocorrelation in 40 ms frames with a 10 ms hop, 70 to 400 Hz.
2. `segment_syllables`: voiced runs separated by pauses. If the count differs from the target,
   the voiced span is cut into equal parts.
3. `classify_tone`: the contour in semitones against the speaker median, reduced to 10 points,
   then fixed rules for level (1), rising (2), dipping or low (3) and falling (4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP_S = 0.01
FRAME_S = 0.04


@dataclass(frozen=True)
class ToneResult:
    tone: int | None
    start_st: float
    end_st: float
    min_st: float
    min_pos: float
    voiced_frames: int


def f0_track(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0,
             voicing: float = 0.5) -> np.ndarray:
    """F0 in Hz for each 10 ms hop. 0 marks an unvoiced frame."""
    n, hop = int(sr * FRAME_S), int(sr * HOP_S)
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    out = []
    peak_all = np.abs(x).max() if len(x) else 0.0
    for start in range(0, max(0, len(x) - n) + 1, hop):
        f = x[start:start + n].astype(np.float64)
        f = f - f.mean()
        if len(f) < n or np.abs(f).max() < 0.05 * peak_all:
            out.append(0.0)
            continue
        ac = np.correlate(f, f, mode="full")[n - 1:]
        if ac[0] <= 0:
            out.append(0.0)
            continue
        ac = ac / ac[0]
        seg = ac[lag_min:lag_max]
        if len(seg) == 0:
            out.append(0.0)
            continue
        k = int(np.argmax(seg))
        if seg[k] < voicing:
            out.append(0.0)
            continue
        lag = lag_min + k
        if 0 < k < len(seg) - 1:  # parabolic interpolation
            a, b, c = seg[k - 1], seg[k], seg[k + 1]
            denom = a - 2 * b + c
            if denom != 0:
                lag = lag + 0.5 * (a - c) / denom
        out.append(sr / lag)
    return np.array(out)


def segment_syllables(f0: np.ndarray, n_expected: int, min_gap: int = 3, min_len: int = 5) -> list[tuple[int, int]]:
    voiced = f0 > 0
    runs: list[list[int]] = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = [r for r in merged if r[1] - r[0] >= min_len]
    if len(merged) == n_expected or not merged or n_expected <= 0:
        return [(a, b) for a, b in merged]
    a, b = merged[0][0], merged[-1][1]
    edges = np.linspace(a, b, n_expected + 1).astype(int)
    return [(int(edges[k]), int(edges[k + 1])) for k in range(n_expected)]


def classify_tone(contour_st: np.ndarray) -> ToneResult:
    c = contour_st[np.isfinite(contour_st)]
    if len(c) < 4:
        return ToneResult(None, 0.0, 0.0, 0.0, 0.0, len(c))
    pts = np.interp(np.linspace(0, len(c) - 1, 10), np.arange(len(c)), c)
    pts = np.convolve(np.pad(pts, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    start, end = float(pts[:2].mean()), float(pts[-2:].mean())
    k = int(np.argmin(pts))
    lo, pos = float(pts[k]), k / 9
    slope = end - start
    if start - lo > 1.5 and end - lo > 1.5 and 0.2 <= pos <= 0.8:
        tone = 3
    elif slope > 2.5:
        tone = 2
    elif slope < -2.5:
        tone = 4
    else:
        tone = 1 if pts.mean() > -1.5 else 3
    return ToneResult(tone, round(start, 2), round(end, 2), round(lo, 2), round(pos, 2), len(c))


def tones_from_audio(x: np.ndarray, sr: int, n_syllables: int) -> list[ToneResult]:
    f0 = f0_track(x, sr)
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return [ToneResult(None, 0, 0, 0, 0, 0)] * n_syllables
    median = np.median(voiced)
    st = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1e-6) / median), np.nan)
    segs = segment_syllables(f0, n_syllables)
    results = [classify_tone(st[a:b]) for a, b in segs]
    results += [ToneResult(None, 0, 0, 0, 0, 0)] * (n_syllables - len(results))
    return results[:n_syllables]


TONE_SHAPES = {  # semitones against the speaker median, at start, middle and end
    1: (3.0, 3.0, 3.0),
    2: (-1.0, 1.0, 4.5),
    3: (-1.0, -6.0, -3.0),
    4: (5.0, 1.0, -5.0),
    5: (0.0, -0.5, -1.0),
}


def synth_tones(tones: list[int], base_hz: float = 160.0, sr: int = 16_000, syll_s: float = 0.3,
                gap_s: float = 0.08, rng: np.random.Generator | None = None, noise: float = 0.01,
                range_scale: float = 1.0) -> np.ndarray:
    """A hummed syllable for each tone, with harmonics, pauses, jitter and noise (tests and demo).

    `range_scale` < 1 compresses the pitch movement, as in fast or flat speech.
    """
    rng = rng or np.random.default_rng(0)
    parts = [np.zeros(int(sr * gap_s))]
    for t in tones:
        n = int(sr * syll_s * rng.uniform(0.85, 1.15))
        a, m, b = (range_scale * v + rng.normal(0, 0.4) for v in TONE_SHAPES[t])
        tt = np.linspace(0, 1, n)
        st = np.interp(tt, [0, 0.5, 1], [a, m, b])
        f = base_hz * 2 ** (st / 12)
        phase = 2 * np.pi * np.cumsum(f) / sr
        wav = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
        env = np.minimum(1, np.minimum(tt, 1 - tt) * 12)
        parts.append(0.3 * wav * env)
        parts.append(np.zeros(int(sr * gap_s)))
    x = np.concatenate(parts)
    return (x + rng.normal(0, noise, len(x))).astype(np.float32)
