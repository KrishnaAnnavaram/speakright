"""G2P, pinyin, alignment, scores and the error taxonomy (problems 4, 5 and 6)."""
import pytest

from speakright.align import align, error_rate, phone_sub_cost, syllable_sub_cost
from speakright.g2p_en import EnglishG2P, G2PError, letter_to_sound
from speakright.phones import PhoneError, ipa_to_arpabet, parse_phones, phone_class
from speakright.pinyin import (PinyinError, apply_sandhi, hanzi_to_pinyin, parse_pinyin, split_syllable,
                               target_syllables, to_tone3)
from speakright.samples import ENGLISH, MANDARIN
from speakright.scoring import Score, band, english_score, mandarin_score
from speakright.taxonomy import classify_english, classify_mandarin_parts


def test_parse_phones_strips_stress_and_rejects_unknown():
    assert parse_phones("DH AH0 K AE1 T") == ["DH", "AH", "K", "AE", "T"]
    with pytest.raises(PhoneError):
        parse_phones("DH QQ")
    assert phone_class("TH") == "fricative" and phone_class("IY") == "vowel"


def test_ipa_map():
    assert ipa_to_arpabet(["θ", "ɪ", "ŋ", "k", "?"]) == ["TH", "IH", "NG", "K"]


def test_g2p_builtin_rules_and_cmudict_file(tmp_path):
    g = EnglishG2P(allow_rules=True)
    assert g.word("three").phones == ("TH", "R", "IY") and not g.word("three").approximate
    approx = g.word("thrush")
    assert approx.approximate and approx.phones[0] == "TH"
    assert letter_to_sound("shine") == ["SH", "IH", "N"]
    f = tmp_path / "dict.txt"
    f.write_text(";;; comment\nTHRUSH  TH R AH1 SH\nTHRUSH(1)  TH R UW1 SH\n", encoding="latin-1")
    g2 = EnglishG2P(cmudict_path=str(f))
    assert g2.word("thrush").phones == ("TH", "R", "AH", "SH") and g2.word("thrush").source == "cmudict"
    with pytest.raises(G2PError):
        EnglishG2P(allow_rules=False).word("zyzzyva")
    with pytest.raises(G2PError):
        g.sentence("123 !!")


def test_all_samples_have_known_pronunciations():
    g = EnglishG2P()
    for s in ENGLISH:
        assert not any(w.approximate for w in g.sentence(s)), s
    for s in MANDARIN:
        assert len(hanzi_to_pinyin(s, use_pypinyin=False)) == len(s)


@pytest.mark.parametrize("raw, tone3", [("nǐ", "ni3"), ("hǎo", "hao3"), ("lǜ", "lv4"), ("ma", "ma5"),
                                        ("xiè", "xie4"), ("ni3", "ni3"), ("Zhōng", "zhong1")])
def test_diacritics_to_tone3(raw, tone3):
    assert to_tone3(raw) == tone3


def test_one_canonical_form_for_characters_and_both_pinyin_styles():
    """Problem 5: characters, diacritic pinyin and TONE3 pinyin give the same units."""
    a = target_syllables("你好！", use_pypinyin=False)
    b = target_syllables("nǐ hǎo", use_pypinyin=False)
    c = target_syllables("ni3, hao3", use_pypinyin=False)
    assert a == b == c == ["ni3", "hao3"]
    assert target_syllables("謝謝", use_pypinyin=False) == ["xie4", "xie5"]  # traditional characters
    with pytest.raises(PinyinError):
        hanzi_to_pinyin("龘", use_pypinyin=False)
    with pytest.raises(PinyinError):
        parse_pinyin("n@1")


def test_tone_sandhi():
    assert apply_sandhi(["ni3", "hao3"]) == ["ni2", "hao3"]
    assert apply_sandhi(["wo3", "hen3", "hao3"]) == ["wo2", "hen2", "hao3"]
    assert apply_sandhi(["bu4", "ke4", "qi5"], "不客气") == ["bu2", "ke4", "qi5"]
    assert apply_sandhi(["yi1", "qi3", "qu4"], "一起去") == ["yi4", "qi3", "qu4"]
    assert apply_sandhi(["yi1", "ding4"]) == ["yi2", "ding4"]


def test_split_syllable():
    s = split_syllable("zhong1")
    assert (s.initial, s.final, s.tone) == ("zh", "ong", 1)
    assert split_syllable("ai4").initial == "" and split_syllable("er2").final == "er"


def test_pypinyin_path_when_installed():
    pytest.importorskip("pypinyin")
    assert target_syllables("你好") == ["ni3", "hao3"]


def test_one_insertion_does_not_shift_later_phones():
    """Problem 6: the old zip-by-position compare flagged every phone after an insertion."""
    ref = "TH R IY TH IH N".split()
    hyp = "TH AH R IY TH IH N".split()
    ops = align(ref, hyp, phone_sub_cost)
    assert [o.kind for o in ops].count("ins") == 1
    assert sum(o.kind != "match" for o in ops) == 1
    assert error_rate(ops, len(ref)) == pytest.approx(1 / 6)


def test_alignment_prefers_same_class_substitutions():
    ops = align(["TH", "IH", "N"], ["T", "IH", "N"], phone_sub_cost)
    assert [(o.kind, o.ref, o.hyp) for o in ops][0] == ("sub", "TH", "T")
    assert phone_sub_cost("TH", "T") > phone_sub_cost("TH", "S")
    assert phone_sub_cost("IY", "T") > 1.0
    assert syllable_sub_cost(split_syllable("ni3"), split_syllable("ni2")) == pytest.approx(0.2)


def test_scores_are_deterministic_and_bounded():
    """Problem 4: the score is a formula of measured values, always a float in [0, 100]."""
    assert english_score(0.0, None).value == 100.0
    assert english_score(0.25, [0.5, 1.0]).value == english_score(0.25, [0.5, 1.0]).value == 75.0
    assert english_score(3.0, None).value == 0.0
    assert mandarin_score(0.0, 0.5).value == 80.0
    with pytest.raises(ValueError):
        Score(101.0, "x", 0.0)
    with pytest.raises(ValueError):
        Score("9", "x", 0.0)  # type: ignore[arg-type]
    assert [band(x) for x in (95, 80, 60, 10)] == ["excellent", "good", "fair", "needs practice"]


def test_taxonomy():
    assert classify_english("sub", "TH", "T", False, False) == "th-stopping"
    assert classify_english("sub", "R", "L", False, False) == "r-l-confusion"
    assert classify_english("del", "D", None, True, False) == "final-consonant-deletion"
    assert classify_english("ins", None, "AH", False, True) == "vowel-epenthesis"
    assert classify_mandarin_parts("zh", "z", "ong", "ong", 1, 1) == ["retroflex-loss"]
    assert classify_mandarin_parts("n", "n", "i", "i", 2, 3) == ["tone-2-as-3"]
    assert classify_mandarin_parts("n", "n", "i", "i", 5, 3) == []  # the neutral tone is not scored
