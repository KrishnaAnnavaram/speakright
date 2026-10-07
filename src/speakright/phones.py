"""The English phone set (ARPAbet, no stress marks), phone classes and an IPA to ARPAbet map."""
from __future__ import annotations

VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
CLASSES: dict[str, frozenset[str]] = {
    "stop": frozenset("P B T D K G".split()),
    "fricative": frozenset("F V TH DH S Z SH ZH HH".split()),
    "affricate": frozenset("CH JH".split()),
    "nasal": frozenset("M N NG".split()),
    "liquid": frozenset("L R".split()),
    "glide": frozenset("W Y".split()),
    "vowel": VOWELS,
}
PHONES = frozenset().union(*CLASSES.values())


class PhoneError(ValueError):
    pass


def phone_class(p: str) -> str:
    for name, members in CLASSES.items():
        if p in members:
            return name
    raise PhoneError(f"unknown ARPAbet phone {p!r}")


def strip_stress(p: str) -> str:
    return p.rstrip("012").upper()


def parse_phones(text: str) -> list[str]:
    """Parse 'DH AH0 K AE1 T' into ['DH', 'AH', 'K', 'AE', 'T'] and check each phone."""
    out = [strip_stress(t) for t in text.split()]
    bad = [p for p in out if p not in PHONES]
    if bad:
        raise PhoneError(f"unknown ARPAbet phone(s) {bad}")
    return out


# espeak-style IPA symbols (as phoneme CTC models emit them) to ARPAbet.
IPA_TO_ARPA: dict[str, str] = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G", "f": "F", "v": "V", "θ": "TH",
    "ð": "DH", "s": "S", "z": "Z", "ʃ": "SH", "ʒ": "ZH", "h": "HH", "tʃ": "CH", "dʒ": "JH", "m": "M", "n": "N",
    "ŋ": "NG", "l": "L", "ɹ": "R", "r": "R", "w": "W", "j": "Y", "ɑ": "AA", "ɑː": "AA", "æ": "AE", "ʌ": "AH",
    "ə": "AH", "ɐ": "AH", "ɔ": "AO", "ɔː": "AO", "aʊ": "AW", "aɪ": "AY", "ɛ": "EH", "e": "EH", "ɜː": "ER",
    "ɚ": "ER", "ɝ": "ER", "eɪ": "EY", "ɪ": "IH", "i": "IY", "iː": "IY", "oʊ": "OW", "əʊ": "OW", "ɔɪ": "OY",
    "ʊ": "UH", "u": "UW", "uː": "UW", "ɾ": "T", "ʔ": "T", "ɫ": "L", "ᵻ": "IH", "ɑːɹ": "AA R", "ɔːɹ": "AO R",
    "ɛɹ": "EH R", "ɪɹ": "IH R", "ʊɹ": "UH R", "oː": "OW", "aː": "AA", "o": "OW", "a": "AA",
}


def ipa_to_arpabet(tokens: list[str]) -> list[str]:
    """Map IPA tokens to ARPAbet. Unknown symbols are dropped (they are counted by the caller)."""
    out: list[str] = []
    for tok in tokens:
        mapped = IPA_TO_ARPA.get(tok) or IPA_TO_ARPA.get(tok.replace("ˈ", "").replace("ˌ", ""))
        if mapped:
            out.extend(mapped.split())
    return out
