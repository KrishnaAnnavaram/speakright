"""Evaluation: error-detection precision, recall and F1, score correlation, and tone accuracy.

- `synthetic_english`: simulated learners and a noisy phone recogniser on the built-in sentences.
  It also scores the old word-level method on the same utterances.
- `synthetic_mandarin`: simulated tone and segment errors. The tones go through real audio
  (`synth_tones`) and the pitch classifier.
- `load_speechocean` and `speechocean_eval`: the public speechocean762 corpus with a real recogniser.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .assess import Assessor
from .g2p_en import EnglishG2P
from .pinyin import apply_sandhi, split_syllable, target_syllables
from .pitch import synth_tones
from .recognize import Recognition
from .samples import ENGLISH, MANDARIN
from .simulate import legacy_word_flags, noisy_recognizer, simulate_english, simulate_mandarin


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def _flagged_ref_positions(assessment) -> set[int]:
    out = set()
    last = 0
    for e in assessment.errors:
        if e.unit_index is not None:
            last = e.unit_index
        out.add(last)
    return out


def synthetic_english(n: int = 200, p_err: float = 0.3, noise: float = 0.05, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    g2p = EnglishG2P()
    assessor = Assessor()
    ptp = pfp = pfn = 0
    w_ours = [0, 0, 0]
    w_old = [0, 0, 0]
    ours_scores, human = [], []
    for i in range(n):
        words = g2p.sentence(ENGLISH[i % len(ENGLISH)])
        utt = simulate_english(words, rng, p_err)
        hyp, conf = noisy_recognizer(utt.actual, rng, noise)
        a = assessor.english(" ".join(w.word for w in words), Recognition(tuple(hyp), tuple(conf)))
        flagged = _flagged_ref_positions(a)
        ptp += len(flagged & utt.error_ref_indices)
        pfp += len(flagged - utt.error_ref_indices)
        pfn += len(utt.error_ref_indices - flagged)
        ours_w = {wi for wi, w in enumerate(a.words) if w["errors"] > 0}
        old_w = legacy_word_flags(words, utt.wrong_words, rng)
        for flags, acc in ((ours_w, w_ours), (old_w, w_old)):
            acc[0] += len(flags & utt.wrong_words)
            acc[1] += len(flags - utt.wrong_words)
            acc[2] += len(utt.wrong_words - flags)
        n_ph = sum(len(w.phones) for w in words)
        human.append(float(np.clip(10 - 25 * utt.n_errors / n_ph + rng.normal(0, 0.5), 0, 10)))
        ours_scores.append(a.score.value)
    return {"utterances": n, "p_err": p_err, "recognizer_noise": noise,
            "phone_detection": prf(ptp, pfp, pfn),
            "word_detection": prf(*w_ours), "legacy_word_detection": prf(*w_old),
            "pearson_with_synthetic_human_score": pearson(ours_scores, human)}


def synthetic_mandarin(n: int = 100, p_tone: float = 0.3, p_seg: float = 0.15, seed: int = 0,
                       noise: float = 0.02, range_scale: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    assessor = Assessor(use_pypinyin=False)
    tone_correct = tone_total = 0
    t_tp = t_fp = t_fn = 0
    s_tp = s_fp = s_fn = 0
    confusion = np.zeros((4, 4), dtype=int)
    for i in range(n):
        text = MANDARIN[i % len(MANDARIN)]
        ref = [split_syllable(s) for s in apply_sandhi(target_syllables(text, False), text)]
        spoken, tone_err, seg_err = simulate_mandarin(ref, rng, p_tone, p_seg)
        tones = [s.tone if s.tone != 5 else 1 for s in spoken]
        audio = synth_tones(tones, base_hz=float(rng.uniform(100, 240)), rng=rng, noise=noise,
                            range_scale=range_scale)
        units = tuple(f"{s.initial}{s.final}5" for s in spoken)  # segments only: the tones come from the audio
        a = assessor.mandarin(text, Recognition(units), audio=audio)
        for k, (r, s) in enumerate(zip(ref, spoken)):
            got = split_syllable(a.recognized[k]).tone if k < len(a.recognized) else None
            if s.tone in (1, 2, 3, 4) and got in (1, 2, 3, 4):
                confusion[s.tone - 1, got - 1] += 1
            if r.tone != 5:
                tone_total += 1
                tone_correct += int(got == s.tone)
        flagged_tone = {e.unit_index for e in a.errors if e.kind.startswith("tone")}
        flagged_seg = {e.unit_index for e in a.errors if not e.kind.startswith("tone")}
        real_tone = {k for k in tone_err if ref[k].tone != 5}
        t_tp += len(flagged_tone & real_tone); t_fp += len(flagged_tone - real_tone); t_fn += len(real_tone - flagged_tone)
        s_tp += len(flagged_seg & seg_err); s_fp += len(flagged_seg - seg_err); s_fn += len(seg_err - flagged_seg)
    return {"utterances": n, "range_scale": range_scale, "noise": noise,
            "pitch_tone_accuracy": round(tone_correct / tone_total, 4) if tone_total else None,
            "tone_confusion_rows_spoken_cols_heard": confusion.tolist(),
            "tone_error_detection": prf(t_tp, t_fp, t_fn), "segment_error_detection": prf(s_tp, s_fp, s_fn)}


# ------------------------------------------------------------------ speechocean762
@dataclass
class OceanItem:
    utt_id: str
    text: str
    wav: Path
    total: float
    mispronounced: set[int]   # canonical phone positions with a human-marked error (0-based, utterance level)
    n_phones: int


def load_speechocean(root: str | Path, split: str = "test") -> list[OceanItem]:
    """Read `<root>/resource/scores.json` and `<root>/<split>/wav.scp` of speechocean762."""
    root = Path(root)
    scores = json.loads((root / "resource" / "scores.json").read_text(encoding="utf-8"))
    wavs = {}
    for line in (root / split / "wav.scp").read_text(encoding="utf-8").splitlines():
        if line.strip():
            uid, path = line.split(maxsplit=1)
            wavs[uid] = root / path.strip()
    items = []
    for uid, wav in sorted(wavs.items()):
        s = scores.get(uid)
        if not s:
            continue
        pos, bad = 0, set()
        for w in s.get("words", []):
            phones = w.get("phones", [])
            for m in w.get("mispronunciations", []):
                bad.add(pos + int(m.get("index", 0)))
            pos += len(phones)
        items.append(OceanItem(uid, s.get("text", ""), wav, float(s.get("total", 0)), bad, pos))
    return items


def speechocean_eval(items: list[OceanItem], recognize: Callable[[Path], Recognition],
                     assessor: Assessor | None = None) -> dict:
    assessor = assessor or Assessor()
    tp = fp = fn = 0
    ours, human = [], []
    skipped = 0
    for it in items:
        try:
            a = assessor.english(it.text, recognize(it.wav))
        except Exception:
            skipped += 1
            continue
        flagged = _flagged_ref_positions(a)
        tp += len(flagged & it.mispronounced); fp += len(flagged - it.mispronounced)
        fn += len(it.mispronounced - flagged)
        ours.append(a.score.value)
        human.append(it.total)
    return {"utterances": len(ours), "skipped": skipped, "phone_detection": prf(tp, fp, fn),
            "pearson_with_human_total": pearson(ours, human)}
