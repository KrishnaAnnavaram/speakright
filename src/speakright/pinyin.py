"""Mandarin: one canonical form (TONE3 pinyin, 5 = neutral tone) on both sides.

- `to_tone3` converts diacritic pinyin (nǐ hǎo) to TONE3 (ni3 hao3).
- `hanzi_to_pinyin` converts characters with pypinyin (extra `g2p`) or with a built-in table.
  Traditional characters of the table map to simplified first. Punctuation is removed.
- `apply_sandhi` gives the expected surface tones: 3-3 -> 2-3, 不 before tone 4, 一 before a tone.
- `split_syllable` splits a syllable into initial, final and tone.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_MARKS = {"̄": 1, "́": 2, "̌": 3, "̀": 4}
INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z",
            "c", "s", "y", "w")

WORDS = {"谢谢": "xie4 xie5", "客气": "ke4 qi5", "妈妈": "ma1 ma5", "朋友": "peng2 you5", "东西": "dong1 xi5"}
CHARS = {
    "你": "ni3", "好": "hao3", "谢": "xie4", "我": "wo3", "很": "hen3", "妈": "ma1", "麻": "ma2", "马": "ma3",
    "骂": "ma4", "不": "bu4", "客": "ke4", "气": "qi4", "一": "yi1", "起": "qi3", "去": "qu4", "中": "zhong1",
    "国": "guo2", "四": "si4", "是": "shi4", "十": "shi2", "今": "jin1", "天": "tian1", "吃": "chi1", "饭": "fan4",
    "朋": "peng2", "友": "you3", "东": "dong1", "西": "xi1", "爱": "ai4", "说": "shuo1", "汉": "han4", "语": "yu3",
    "老": "lao3", "师": "shi1", "水": "shui3", "喝": "he1", "茶": "cha2", "早": "zao3", "上": "shang4",
}
TRAD_TO_SIMP = {"謝": "谢", "氣": "气", "媽": "妈", "馬": "马", "罵": "骂", "國": "国", "飯": "饭", "東": "东",
                "說": "说", "漢": "汉", "語": "语", "師": "师", "愛": "爱"}


class PinyinError(ValueError):
    pass


@dataclass(frozen=True)
class Syllable:
    initial: str
    final: str
    tone: int

    @property
    def text(self) -> str:
        return f"{self.initial}{self.final}{self.tone}"


def to_tone3(syllable: str) -> str:
    """'nǐ' -> 'ni3', 'lǜ' -> 'lv4', 'ma' -> 'ma5', 'ni3' -> 'ni3'."""
    s = syllable.strip().lower()
    if re.fullmatch(r"[a-zv]+[1-5]", s):
        return s
    tone = 5
    base = []
    for ch in unicodedata.normalize("NFD", s):
        if ch in _MARKS:
            tone = _MARKS[ch]
        elif ch == "̈":
            if base and base[-1] == "u":
                base[-1] = "v"
        else:
            base.append(ch)
    plain = "".join(base).replace("ü", "v")
    if not re.fullmatch(r"[a-zv]+", plain):
        raise PinyinError(f"not a pinyin syllable: {syllable!r}")
    return f"{plain}{tone}"


def parse_pinyin(text: str) -> list[str]:
    return [to_tone3(t) for t in re.split(r"[\s,.;!?，。！？、]+", text.strip()) if t]


def _pypinyin(text: str) -> list[str] | None:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return None
    return [s for s in lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True) if re.fullmatch(r"[a-zv]+[1-5]", s)]


def hanzi_to_pinyin(text: str, use_pypinyin: bool = True) -> list[str]:
    chars = "".join(TRAD_TO_SIMP.get(c, c) for c in text if "一" <= c <= "鿿")
    if not chars:
        raise PinyinError("the text has no Chinese character")
    if use_pypinyin:
        got = _pypinyin(chars)
        if got:
            return got
    out: list[str] = []
    i = 0
    while i < len(chars):
        if chars[i:i + 2] in WORDS:
            out.extend(WORDS[chars[i:i + 2]].split())
            i += 2
        elif chars[i] in CHARS:
            out.append(CHARS[chars[i]])
            i += 1
        else:
            raise PinyinError(f"no pinyin for {chars[i]!r}; install the extra g2p (pypinyin) or give pinyin")
    return out


def target_syllables(text: str, use_pypinyin: bool = True) -> list[str]:
    """Characters or pinyin (diacritics or TONE3) -> TONE3 syllables (before sandhi)."""
    if any("一" <= c <= "鿿" for c in text):
        return hanzi_to_pinyin(text, use_pypinyin)
    return parse_pinyin(text)


def apply_sandhi(sylls: list[str], chars: str | None = None) -> list[str]:
    """Expected surface tones. `chars` (same length) marks 不 and 一; without it, bu4/yi1 are used."""
    out = list(sylls)
    for i in range(len(out) - 1):
        nxt = int(out[i + 1][-1])
        base = out[i][:-1]
        is_bu = (chars[i] == "不") if chars and len(chars) == len(out) else base == "bu"
        is_yi = (chars[i] == "一") if chars and len(chars) == len(out) else base == "yi" and out[i][-1] == "1"
        if is_bu and nxt == 4:
            out[i] = "bu2"
        elif is_yi and nxt in (1, 2, 3):
            out[i] = "yi4"
        elif is_yi and nxt == 4:
            out[i] = "yi2"
    # third-tone sandhi, right to left inside the phrase: 3 3 -> 2 3, 3 3 3 -> 2 2 3
    for i in range(len(out) - 2, -1, -1):
        if out[i].endswith("3") and sylls[i + 1].endswith("3"):
            out[i] = out[i][:-1] + "2"
    return out


def split_syllable(s: str) -> Syllable:
    s = to_tone3(s)
    body, tone = s[:-1], int(s[-1])
    for ini in INITIALS:
        if body.startswith(ini) and len(body) > len(ini):
            return Syllable(ini, body[len(ini):], tone)
    return Syllable("", body, tone)
