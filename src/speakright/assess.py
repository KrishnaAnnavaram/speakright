"""The assessment pipeline: target -> canonical units, recognised units -> alignment -> errors -> score -> tips.

The LLM tip writer is called at most ONCE for each assessment, after the score is final.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .align import align, error_rate, phone_sub_cost
from .g2p_en import EnglishG2P
from .llm import LLM, SYSTEM, LLMError, parse_tips
from .phones import VOWELS
from .pinyin import Syllable, apply_sandhi, split_syllable, target_syllables
from .pitch import tones_from_audio
from .recognize import Recognition
from .scoring import Score, english_score, mandarin_score
from .taxonomy import TIPS, ErrorItem, classify_english, classify_mandarin_parts

LANGUAGES = ("en", "zh")


class AssessError(ValueError):
    pass


@dataclass
class Assessment:
    language: str
    target: str
    canonical: list[str]
    recognized: list[str]
    words: list[dict]
    ops: list[dict]
    errors: list[ErrorItem]
    score: Score
    tips: list[str]
    tip_source: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["errors"] = [asdict(e) for e in self.errors]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def rule_tips(errors: list[ErrorItem], limit: int = 5) -> list[str]:
    seen: list[str] = []
    for e in errors:
        if e.kind not in seen:
            seen.append(e.kind)
    if not seen:
        return ["No error found. Practise a longer sentence or a faster pace."]
    return [f"{k}: {TIPS.get(k, TIPS['substitution'])}" for k in seen[:limit]]


class Assessor:
    def __init__(self, llm: LLM | None = None, g2p: EnglishG2P | None = None, use_pypinyin: bool = True):
        self.llm = llm
        self.g2p = g2p or EnglishG2P()
        self.use_pypinyin = use_pypinyin

    # ---------------------------------------------------------------- English
    def english(self, target: str, rec: Recognition) -> Assessment:
        words = self.g2p.sentence(target)
        ref: list[str] = []
        word_of: list[int] = []
        last_of_word: set[int] = set()
        for wi, wp in enumerate(words):
            ref.extend(wp.phones)
            word_of.extend([wi] * len(wp.phones))
            last_of_word.add(len(ref) - 1)
        hyp = list(rec.units)
        ops = align(ref, hyp, phone_sub_cost)
        errors: list[ErrorItem] = []
        cur_word = 0
        for op in ops:
            if op.ref_index is not None:
                cur_word = word_of[op.ref_index]
            if op.kind == "match":
                continue
            prev_hyp = hyp[op.hyp_index - 1] if op.hyp_index else None
            kind = classify_english(op.kind, op.ref, op.hyp,
                                    at_word_end=op.ref_index in last_of_word if op.ref_index is not None else False,
                                    after_consonant=prev_hyp is not None and prev_hyp not in VOWELS)
            errors.append(ErrorItem(kind, op.ref or "-", op.hyp or "-", op.ref_index, words[cur_word].word))
        conf = None
        if rec.confidences:
            conf = [rec.confidences[op.hyp_index] for op in ops if op.kind == "match"] or [0.0]
        score = english_score(error_rate(ops, len(ref)), conf)
        word_rows = []
        for wi, wp in enumerate(words):
            n_err = sum(e.word == wp.word for e in errors)
            word_rows.append({"word": wp.word, "phones": " ".join(wp.phones), "errors": n_err,
                              "approximate": wp.approximate})
        notes = [f"approximate pronunciation for {w.word!r} (letter-to-sound rules)" for w in words if w.approximate]
        if rec.dropped:
            notes.append(f"{rec.dropped} recognised symbol(s) had no ARPAbet mapping")
        return self._finish("en", target, ref, hyp, word_rows, ops, errors, score, notes)

    # --------------------------------------------------------------- Mandarin
    def mandarin(self, target: str, rec: Recognition, audio: np.ndarray | None = None, sr: int = 16_000) -> Assessment:
        raw = target_syllables(target, self.use_pypinyin)
        chars = "".join(c for c in target if "一" <= c <= "鿿") or None
        ref_s = apply_sandhi(raw, chars)
        ref = [split_syllable(s) for s in ref_s]
        hyp = [split_syllable(s) for s in rec.units]
        notes = []
        if audio is not None and len(hyp):
            tones = tones_from_audio(audio, sr, len(hyp))
            hyp = [Syllable(h.initial, h.final, t.tone if t.tone is not None else h.tone) for h, t in zip(hyp, tones)]
            notes.append("tones measured from the pitch of the audio")
        seg_cost = lambda a, b: 0.5 * (a.initial != b.initial) + 0.5 * (a.final != b.final)  # noqa: E731
        ops = align(ref, hyp, seg_cost)
        errors: list[ErrorItem] = []
        seg_errors = 0
        tone_total = tone_ok = 0
        for op in ops:
            r, h = op.ref, op.hyp
            if r is not None and r.tone != 5:
                tone_total += 1
                tone_ok += int(h is not None and h.tone == r.tone)
            if op.kind == "del":
                seg_errors += 1
                errors.append(ErrorItem("syllable-deletion", r.text, "-", op.ref_index, r.text))
            elif op.kind == "ins":
                seg_errors += 1
                errors.append(ErrorItem("syllable-insertion", "-", h.text, None, h.text))
            else:
                kinds = classify_mandarin_parts(r.initial, h.initial, r.final, h.final, r.tone, h.tone)
                if op.kind == "sub":
                    seg_errors += 1
                for k in kinds:
                    errors.append(ErrorItem(k, r.text, h.text, op.ref_index, r.text))
        if any(a != b for a, b in zip(raw, ref_s)):
            notes.append("tone sandhi applied: " + " ".join(raw) + " -> " + " ".join(ref_s))
        tone_acc = tone_ok / tone_total if tone_total else None
        score = mandarin_score(seg_errors / len(ref) if ref else 1.0, tone_acc)
        word_rows = [{"word": s.text, "phones": f"{s.initial} {s.final} {s.tone}".strip(),
                      "errors": sum(e.unit_index == i for e in errors), "approximate": False} for i, s in enumerate(ref)]
        return self._finish("zh", target, [s.text for s in ref], [s.text for s in hyp], word_rows, ops, errors,
                            score, notes)

    # ----------------------------------------------------------------- common
    def _finish(self, lang, target, ref, hyp, word_rows, ops, errors, score, notes) -> Assessment:
        def show(u):
            return u.text if hasattr(u, "text") else u
        op_rows = [{"op": o.kind, "expected": show(o.ref) if o.ref is not None else "-",
                    "actual": show(o.hyp) if o.hyp is not None else "-"} for o in ops]
        result = Assessment(lang, target, ref, hyp, word_rows, op_rows, errors, score, rule_tips(errors), "rules",
                            notes)
        if self.llm is not None and errors:
            payload = json.dumps({"language": lang, "target": target, "score": asdict(score),
                                  "errors": [asdict(e) for e in errors[:20]]}, ensure_ascii=False)
            try:
                result.tips = parse_tips(self.llm.complete(SYSTEM, payload))
                result.tip_source = "llm"
            except (LLMError, Exception) as exc:  # any LLM failure keeps the rule tips
                result.notes.append(f"LLM tips not used: {type(exc).__name__}")
        return result

    def assess(self, language: str, target: str, rec: Recognition, audio: np.ndarray | None = None,
               sr: int = 16_000) -> Assessment:
        if language not in LANGUAGES:
            raise AssessError(f"language must be one of {LANGUAGES}")
        if not rec.units:
            raise AssessError("the recogniser found no sound; record again, closer to the microphone")
        return self.english(target, rec) if language == "en" else self.mandarin(target, rec, audio, sr)


def to_markdown(a: Assessment) -> str:
    lines = [f"## Score: {a.score.value:.1f} / 100 ({a.score.band})", "",
             f"Target: {a.target}", "", "| Unit | Expected | Errors |", "|---|---|---|"]
    lines += [f"| {w['word']} | {w['phones']} | {w['errors']} |" for w in a.words]
    if a.errors:
        lines += ["", "### Errors", ""]
        lines += [f"- {e.word}: {e.kind} (expected {e.expected}, heard {e.actual})" for e in a.errors]
    lines += ["", f"### Tips ({a.tip_source})", ""] + [f"- {t}" for t in a.tips]
    if a.notes:
        lines += ["", "### Notes", ""] + [f"- {n}" for n in a.notes]
    return "\n".join(lines)
