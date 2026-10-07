"""Named error types for aligned differences, with a fixed tip for each type."""
from __future__ import annotations

from dataclasses import dataclass

from .phones import VOWELS

EN_SUBS: dict[tuple[str, str], str] = {
    ("TH", "T"): "th-stopping", ("TH", "S"): "th-fronting", ("TH", "F"): "th-fronting",
    ("DH", "D"): "th-stopping", ("DH", "Z"): "th-fronting", ("DH", "V"): "th-fronting",
    ("R", "L"): "r-l-confusion", ("L", "R"): "r-l-confusion",
    ("V", "W"): "v-w-confusion", ("W", "V"): "v-w-confusion", ("V", "B"): "v-b-confusion", ("B", "V"): "v-b-confusion",
    ("IY", "IH"): "vowel-length", ("IH", "IY"): "vowel-length", ("UW", "UH"): "vowel-length",
    ("UH", "UW"): "vowel-length", ("Z", "S"): "final-devoicing", ("D", "T"): "final-devoicing",
    ("V", "F"): "final-devoicing", ("SH", "S"): "sh-s-confusion", ("S", "SH"): "sh-s-confusion",
    ("N", "NG"): "n-ng-confusion", ("NG", "N"): "n-ng-confusion",
}
ZH_INITIALS: dict[tuple[str, str], str] = {
    ("zh", "z"): "retroflex-loss", ("ch", "c"): "retroflex-loss", ("sh", "s"): "retroflex-loss",
    ("z", "zh"): "retroflex-added", ("c", "ch"): "retroflex-added", ("s", "sh"): "retroflex-added",
    ("n", "l"): "n-l-confusion", ("l", "n"): "n-l-confusion", ("r", "l"): "r-l-confusion",
    ("q", "ch"): "palatal-confusion", ("x", "sh"): "palatal-confusion", ("j", "zh"): "palatal-confusion",
}

TIPS: dict[str, str] = {
    "th-stopping": "Put the tip of the tongue between the teeth and let air flow. Do not stop the air.",
    "th-fronting": "Use the tongue tip, not the lip or the teeth ridge, for 'th'.",
    "r-l-confusion": "For 'r', curl the tongue back and do not touch the roof. For 'l', touch the ridge behind the teeth.",
    "v-w-confusion": "For 'v', touch the lower lip with the upper teeth. For 'w', round the lips.",
    "v-b-confusion": "For 'v', let air pass between the teeth and the lip. Do not close the lips.",
    "vowel-length": "Make the long vowel (as in 'sheep') longer and tenser than the short vowel (as in 'ship').",
    "final-devoicing": "Keep the voice on at the end of the word.",
    "sh-s-confusion": "For 'sh', round the lips and pull the tongue back a little.",
    "n-ng-confusion": "For 'ng', raise the back of the tongue. For 'n', use the tongue tip.",
    "final-consonant-deletion": "Say the last consonant of the word clearly.",
    "vowel-epenthesis": "Do not add a vowel after the consonant.",
    "phone-deletion": "Say each sound of the word.",
    "phone-insertion": "Remove the extra sound.",
    "substitution": "Listen to the reference and copy the sound.",
    "retroflex-loss": "For zh, ch and sh, curl the tongue tip up and back.",
    "retroflex-added": "For z, c and s, keep the tongue tip flat behind the teeth.",
    "n-l-confusion": "For 'n', let air pass through the nose. For 'l', let air pass the sides of the tongue.",
    "palatal-confusion": "For j, q and x, keep the tongue tip down behind the lower teeth.",
    "final-error": "Check the vowel and the nasal ending (-n or -ng) of the syllable.",
    "tone-error": "Copy the pitch shape of the tone: 1 high and level, 2 rising, 3 low or dipping, 4 falling.",
    "tone-3-as-2": "Tone 3 starts low and dips. Do not rise as in tone 2.",
    "tone-2-as-3": "Tone 2 rises from the middle to high. Do not dip first.",
    "syllable-deletion": "Say each syllable.",
    "syllable-insertion": "Remove the extra syllable.",
}


@dataclass(frozen=True)
class ErrorItem:
    kind: str           # error type, a key of TIPS
    expected: str
    actual: str
    unit_index: int | None
    word: str


def classify_english(kind: str, ref: str | None, hyp: str | None, at_word_end: bool, after_consonant: bool) -> str:
    if kind == "sub":
        return EN_SUBS.get((ref, hyp), "substitution")
    if kind == "del":
        return "final-consonant-deletion" if at_word_end and ref not in VOWELS else "phone-deletion"
    return "vowel-epenthesis" if hyp in VOWELS and after_consonant else "phone-insertion"


def classify_mandarin_parts(ref_init: str, hyp_init: str, ref_final: str, hyp_final: str,
                            ref_tone: int, hyp_tone: int | None) -> list[str]:
    out = []
    if ref_init != hyp_init:
        out.append(ZH_INITIALS.get((ref_init, hyp_init), "substitution"))
    if ref_final != hyp_final:
        out.append("final-error")
    if ref_tone != 5 and hyp_tone is not None and ref_tone != hyp_tone:
        out.append({(3, 2): "tone-3-as-2", (2, 3): "tone-2-as-3"}.get((ref_tone, hyp_tone), "tone-error"))
    return out
