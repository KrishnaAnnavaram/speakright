"""Phone recognisers. They return what the speaker SAID (phones), not the words that a language model expects.

- `CtcPhoneRecognizer`: a wav2vec2 phoneme CTC checkpoint (extra `recognizer`), greedy decode, a
  confidence for each phone (the mean top-class posterior over its frames, a GOP-like value).
- `ScriptedRecognizer`: returns a given sequence. The tests, the demo and `--phones` use it.
`get_recognizer` keeps one loaded model for each name in the process, so the model loads once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np

from .phones import ipa_to_arpabet

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


@dataclass(frozen=True)
class Recognition:
    units: tuple[str, ...]
    confidences: tuple[float, ...] | None = None
    source: str = "scripted"
    dropped: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


class Recognizer(Protocol):
    name: str

    def recognize(self, x: np.ndarray, sr: int) -> Recognition: ...


class ScriptedRecognizer:
    name = "scripted"

    def __init__(self, units: list[str], confidences: list[float] | None = None):
        if confidences is not None and len(confidences) != len(units):
            raise ValueError("one confidence for each unit")
        self.units = tuple(units)
        self.confidences = tuple(confidences) if confidences is not None else None

    def recognize(self, x: np.ndarray | None = None, sr: int = 16_000) -> Recognition:
        return Recognition(self.units, self.confidences, "scripted")


class CtcPhoneRecognizer:  # pragma: no cover - needs torch, transformers and a model download
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).eval().to(device)
        self.device = device
        self.name = f"ctc:{model_name}"

    def recognize(self, x: np.ndarray, sr: int) -> Recognition:
        inputs = self.processor(x, sampling_rate=sr, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
        probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
        ids = probs.argmax(axis=1)
        tok = self.processor.tokenizer
        special = {tok.pad_token_id}
        units, confs = [], []
        prev = None
        run: list[float] = []
        for t, i in enumerate(ids):
            if i != prev:
                if prev is not None and prev not in special and run:
                    units.append(tok.convert_ids_to_tokens(int(prev)))
                    confs.append(float(np.mean(run)))
                run = []
                prev = i
            run.append(float(probs[t, i]))
        if prev is not None and prev not in special and run:
            units.append(tok.convert_ids_to_tokens(int(prev)))
            confs.append(float(np.mean(run)))
        phones, pconf, dropped = [], [], 0
        for u, c in zip(units, confs):
            mapped = ipa_to_arpabet([u])
            if not mapped:
                dropped += 1
            for p in mapped:
                phones.append(p)
                pconf.append(c)
        return Recognition(tuple(phones), tuple(pconf), self.name, dropped)


@lru_cache(maxsize=2)
def get_recognizer(model_name: str = DEFAULT_MODEL, device: str = "cpu") -> CtcPhoneRecognizer:
    return CtcPhoneRecognizer(model_name, device)
