"""English grapheme-to-phoneme: text -> canonical ARPAbet phones for each word.

Order of sources:
1. A CMUdict file (`SPEAKRIGHT_CMUDICT`) or the `cmudict` package (extra `g2p`), if available.
2. The built-in lexicon below (all words of the built-in practice sentences).
3. A small letter-to-sound rule set. Its result is marked `approximate=True`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .phones import PHONES, strip_stress

BUILTIN_LEXICON: dict[str, str] = {
    "a": "AH", "and": "AE N D", "by": "B AY", "call": "K AO L", "coffee": "K AO F IY", "cup": "K AH P",
    "good": "G UH D", "hello": "HH AH L OW", "i": "AY", "is": "IH Z", "light": "L AY T", "like": "L AY K",
    "lorry": "L AO R IY", "morning": "M AO R N IH NG", "nice": "N AY S", "not": "N AA T", "of": "AH V",
    "please": "P L IY Z", "red": "R EH D", "rice": "R AY S", "right": "R AY T", "sea": "S IY",
    "sells": "S EH L Z", "she": "SH IY", "sheep": "SH IY P", "shells": "SH EH L Z", "ship": "SH IH P",
    "shore": "SH AO R", "stella": "S T EH L AH", "thank": "TH AE NG K", "the": "DH AH", "thieves": "TH IY V Z",
    "thin": "TH IH N", "this": "DH IH S", "thought": "TH AO T", "thoughts": "TH AO T S", "thousand": "TH AW Z AH N D",
    "three": "TH R IY", "today": "T AH D EY", "very": "V EH R IY", "village": "V IH L IH JH",
    "vincent": "V IH N S AH N T", "visit": "V IH Z IH T", "we": "W IY", "weather": "W EH DH ER", "will": "W IH L",
    "with": "W IH DH", "world": "W ER L D", "would": "W UH D", "yellow": "Y EH L OW", "you": "Y UW",
    "think": "TH IH NG K", "they": "DH EY", "there": "DH EH R", "water": "W AO T ER", "light's": "L AY T S",
    "lead": "L IY D", "read": "R IY D", "leaf": "L IY F", "reef": "R IY F", "vest": "V EH S T", "west": "W EH S T",
}

_DIGRAPHS = [("tch", "CH"), ("th", "TH"), ("sh", "SH"), ("ch", "CH"), ("ng", "NG"), ("ph", "F"), ("ck", "K"),
             ("wh", "W"), ("qu", "K W"), ("ee", "IY"), ("ea", "IY"), ("oo", "UW"), ("ai", "EY"), ("ay", "EY"),
             ("oa", "OW"), ("ow", "OW"), ("ou", "AW"), ("oi", "OY"), ("oy", "OY"), ("igh", "AY"), ("er", "ER"),
             ("ir", "ER"), ("ur", "ER"), ("ar", "AA R"), ("or", "AO R")]
_SINGLE = {"a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G", "h": "HH", "i": "IH",
           "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "o": "AA", "p": "P", "q": "K", "r": "R", "s": "S",
           "t": "T", "u": "AH", "v": "V", "w": "W", "x": "K S", "y": "IY", "z": "Z"}


class G2PError(ValueError):
    pass


@dataclass(frozen=True)
class WordPhones:
    word: str
    phones: tuple[str, ...]
    approximate: bool = False
    source: str = "builtin"


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def letter_to_sound(word: str) -> list[str]:
    """Rough rules for a word that no lexicon has. The result is only an approximation."""
    w = word.lower().replace("'", "")
    if len(w) > 2 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]  # silent final e
    out: list[str] = []
    i = 0
    while i < len(w):
        for g, ph in _DIGRAPHS:
            if w.startswith(g, i):
                out.extend(ph.split())
                i += len(g)
                break
        else:
            ch = w[i]
            if ch == "y" and i == 0:
                out.append("Y")
            elif ch in _SINGLE:
                out.extend(_SINGLE[ch].split())
            i += 1
    collapsed = [p for j, p in enumerate(out) if j == 0 or p != out[j - 1]]
    return collapsed


@lru_cache(maxsize=4)
def _load_cmudict(path: str | None) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    if path:
        for line in Path(path).read_text(encoding="latin-1").splitlines():
            if not line or line.startswith(";;;"):
                continue
            head, _, pron = line.partition(" ")
            word = re.sub(r"\(\d+\)$", "", head).lower()
            entries.setdefault(word, [strip_stress(p) for p in pron.split()])
        return entries
    try:
        import cmudict  # optional extra
    except ImportError:
        return {}
    for word, prons in cmudict.dict().items():
        if prons:
            entries[word] = [strip_stress(p) for p in prons[0]]
    return entries


class EnglishG2P:
    def __init__(self, cmudict_path: str | None = None, allow_rules: bool = True):
        self.cmu = _load_cmudict(cmudict_path or os.environ.get("SPEAKRIGHT_CMUDICT") or None)
        self.allow_rules = allow_rules

    def word(self, word: str) -> WordPhones:
        w = word.lower()
        if w in self.cmu:
            return WordPhones(w, tuple(self.cmu[w]), source="cmudict")
        if w in BUILTIN_LEXICON:
            return WordPhones(w, tuple(BUILTIN_LEXICON[w].split()))
        if not self.allow_rules:
            raise G2PError(f"no pronunciation for {word!r}")
        return WordPhones(w, tuple(letter_to_sound(w)), approximate=True, source="rules")

    def sentence(self, text: str) -> list[WordPhones]:
        words = words_of(text)
        if not words:
            raise G2PError("the target text has no English word")
        out = [self.word(w) for w in words]
        for wp in out:
            bad = [p for p in wp.phones if p not in PHONES]
            if bad:
                raise G2PError(f"bad phones {bad} for {wp.word!r}")
        return out
