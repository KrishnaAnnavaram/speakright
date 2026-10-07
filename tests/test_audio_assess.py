"""Audio, pitch, the assessment pipeline and the LLM tip writer (problems 1, 2, 3 and 5)."""
import json

import numpy as np
import pytest

from speakright.assess import AssessError, Assessor, to_markdown
from speakright.audio import AudioError, check, prepare, read_wav, to_16k, vad_trim, write_wav
from speakright.g2p_en import EnglishG2P
from speakright.llm import FakeLLM, LLMError, parse_tips
from speakright.pitch import classify_tone, f0_track, segment_syllables, synth_tones, tones_from_audio
from speakright.recognize import Recognition, ScriptedRecognizer


def _tone(sr=16_000, f=200.0, s=1.0, amp=0.3):
    t = np.arange(int(sr * s)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_wav_round_trip_resample_and_trim(tmp_path):
    x = np.concatenate([np.zeros(8000), _tone(sr=8000), np.zeros(8000)]).astype(np.float32)
    p = write_wav(tmp_path / "a.wav", x, 8000)
    y, sr = read_wav(p)
    assert sr == 8000 and np.allclose(x, y, atol=1e-3)
    z = to_16k(y, sr)
    assert len(z) == 2 * len(y)
    trimmed = vad_trim(z, 16_000)
    assert 0.9 * 16_000 < len(trimmed) < 1.5 * 16_000
    audio, c = prepare(p.read_bytes())
    assert c.ok and len(audio) == len(trimmed)


def test_audio_checks():
    assert "too short" in check(_tone(s=0.1), 16_000).problems[0]
    assert any("too quiet" in p for p in check(_tone(amp=0.001), 16_000).problems)
    assert any("clipped" in p for p in check(np.clip(_tone(amp=3), -1, 1), 16_000).problems)
    with pytest.raises(AudioError):
        read_wav(b"not a wav")


def test_f0_of_a_sine():
    f0 = f0_track(_tone(f=220.0), 16_000)
    voiced = f0[f0 > 0]
    assert len(voiced) > 80 and abs(np.median(voiced) - 220.0) < 3.0


@pytest.mark.parametrize("tone", [1, 2, 3, 4])
def test_tone_classes_from_synthetic_audio(tone):
    x = synth_tones([1, tone, 4], base_hz=150.0, rng=np.random.default_rng(tone))
    assert tones_from_audio(x, 16_000, 3)[1].tone == tone


def test_tone_rules_and_segment_fallback():
    assert classify_tone(np.linspace(-4, 4, 20)).tone == 2
    assert classify_tone(np.linspace(4, -4, 20)).tone == 4
    assert classify_tone(np.array([0, -3, -5, -5, -3, -1, 0, 0.5] * 2)[:16]).tone in (3, 2)
    assert classify_tone(np.array([1.0, 2.0])).tone is None
    f0 = np.array([0] * 3 + [200] * 40 + [0] * 3)
    assert segment_syllables(f0, 2) == [(3, 23), (23, 43)]  # one voiced run cut into 2 equal parts


EN = "three thin thieves"


def test_english_errors_come_from_the_phones():
    """Problem 1: an error in the heard phones gives an error, also when the word is right."""
    heard = "T R IY TH IH N TH IY V Z".split()
    a = Assessor().assess("en", EN, Recognition(tuple(heard)))
    assert [(e.kind, e.word) for e in a.errors] == [("th-stopping", "three")]
    assert a.score.value == pytest.approx(90.0) and a.tip_source == "rules"
    md = to_markdown(a)
    assert "th-stopping" in md and "90.0" in md


def test_english_insertion_and_deletion():
    heard = "TH R IY TH IH N TH IY V".split() + ["AH"]
    a = Assessor().assess("en", EN, Recognition(tuple(heard)))
    kinds = sorted(e.kind for e in a.errors)
    assert "final-devoicing" in kinds or "final-consonant-deletion" in kinds or "substitution" in kinds


def test_perfect_english_with_confidences():
    g = EnglishG2P()
    phones = [p for w in g.sentence(EN) for p in w.phones]
    a = Assessor().assess("en", EN, Recognition(tuple(phones), tuple([0.9] * len(phones))))
    assert a.errors == [] and a.score.value == pytest.approx(98.0) and a.score.gop == 0.9


def test_mandarin_tone_from_audio_and_sandhi():
    """Problem 5: one format on both sides. The heard tones come from the pitch, not from text."""
    a = Assessor(use_pypinyin=False)
    ok = a.assess("zh", "你好", Recognition(("ni5", "hao5")), synth_tones([2, 3], base_hz=170.0))
    assert ok.errors == [] and ok.canonical == ["ni2", "hao3"] and ok.score.value == 100.0
    bad = a.assess("zh", "nǐ hǎo", Recognition(("ni5", "hao5")), synth_tones([3, 3], base_hz=170.0))
    assert [e.kind for e in bad.errors] == ["tone-2-as-3"] and bad.score.tone_accuracy == 0.5


def test_mandarin_segment_errors_without_audio():
    a = Assessor(use_pypinyin=False).assess("zh", "四是四十是十", Recognition(tuple("si4 si4 si4 shi2 shi4 shi2".split())))
    assert [e.kind for e in a.errors] == ["retroflex-loss"]
    assert a.score.error_rate == pytest.approx(1 / 6, abs=1e-4)


def test_llm_is_called_once_and_never_sets_the_score():
    """Problem 2: one LLM call for each assessment, after the score."""
    llm = FakeLLM(json.dumps({"tips": ["Bite the tongue lightly for th.", "Score: 10/10"]}))
    heard = "T R IY TH IH N TH IY V Z".split()
    a = Assessor(llm=llm).assess("en", EN, Recognition(tuple(heard)))
    assert llm.calls == 1 and a.tip_source == "llm" and a.score.value == pytest.approx(90.0)


def test_no_llm_and_bad_llm_replies_fall_back_to_rules():
    """Problem 3: no key is a normal path. A bad reply never breaks the result."""
    heard = "T R IY TH IH N TH IY V Z".split()
    for llm in (None, FakeLLM("not json"), FakeLLM(json.dumps({"tips": []})), FakeLLM(json.dumps({"tips": ["x" * 300]}))):
        a = Assessor(llm=llm).assess("en", EN, Recognition(tuple(heard)))
        assert a.tip_source == "rules" and a.tips and a.score.value == pytest.approx(90.0)
    with pytest.raises(LLMError):
        parse_tips('{"tips": "one"}')


def test_no_llm_call_without_errors():
    llm = FakeLLM(json.dumps({"tips": ["x"]}))
    phones = [p for w in EnglishG2P().sentence(EN) for p in w.phones]
    Assessor(llm=llm).assess("en", EN, Recognition(tuple(phones)))
    assert llm.calls == 0


def test_bad_inputs():
    with pytest.raises(AssessError):
        Assessor().assess("fr", "bonjour", Recognition(("B",)))
    with pytest.raises(AssessError):
        Assessor().assess("en", EN, Recognition(()))
    with pytest.raises(ValueError):
        ScriptedRecognizer(["A"], [0.1, 0.2])
    assert ScriptedRecognizer(["TH"]).recognize().units == ("TH",)
