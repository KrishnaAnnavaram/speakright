"""Evaluation, privacy, settings, recogniser cache, CLI and UI (problems 1, 7, 8, 10 and 11)."""
import json
import os
import time

import numpy as np
import pytest

from speakright import recognize as rec_mod
from speakright.audio import write_wav
from speakright.cli import main
from speakright.config import ConfigError, Settings
from speakright.evaluate import (load_speechocean, pearson, prf, speechocean_eval, synthetic_english,
                                 synthetic_mandarin)
from speakright.g2p_en import EnglishG2P
from speakright.pitch import synth_tones
from speakright.privacy import purge, store
from speakright.recognize import Recognition


def test_prf_and_pearson():
    assert prf(8, 2, 2) == {"precision": 0.8, "recall": 0.8, "f1": 0.8, "tp": 8, "fp": 2, "fn": 2}
    assert prf(0, 0, 0)["f1"] == 0.0
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0 and pearson([1, 1, 1], [1, 2, 3]) == 0.0


def test_phone_method_beats_the_word_method_on_labelled_errors():
    """Problems 1 and 11: labelled errors, precision, recall, F1 and a correlation."""
    r = synthetic_english(n=40, seed=1)
    assert r["word_detection"]["recall"] > 0.9
    assert r["legacy_word_detection"]["recall"] < 0.4
    assert r["phone_detection"]["f1"] > 0.7 and r["pearson_with_synthetic_human_score"] > 0.6


def test_synthetic_mandarin_eval():
    r = synthetic_mandarin(n=20, seed=2)
    assert r["pitch_tone_accuracy"] > 0.9 and r["tone_error_detection"]["recall"] > 0.8


def test_speechocean_loader_and_eval(tmp_path):
    (tmp_path / "resource").mkdir()
    (tmp_path / "test").mkdir()
    scores = {
        "u1": {"text": "WE CALL", "total": 9, "words": [
            {"text": "WE", "phones": ["W", "IY0"], "mispronunciations": []},
            {"text": "CALL", "phones": ["K", "AO0", "L"],
             "mispronunciations": [{"canonical-phone": "L", "index": 2, "pronounced-phone": "R"}]}]},
        "u2": {"text": "WE WILL", "total": 4, "words": [
            {"text": "WE", "phones": ["W", "IY0"], "mispronunciations": [{"index": 0}]},
            {"text": "WILL", "phones": ["W", "IH0", "L"], "mispronunciations": []}]},
        "u3": {"text": "PLEASE CALL", "total": 10, "words": [
            {"text": "PLEASE", "phones": ["P", "L", "IY0", "Z"], "mispronunciations": []},
            {"text": "CALL", "phones": ["K", "AO0", "L"], "mispronunciations": []}]},
    }
    (tmp_path / "resource" / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (tmp_path / "test" / "wav.scp").write_text("u1 WAVE/u1.wav\nu2 WAVE/u2.wav\nu3 WAVE/u3.wav\n", encoding="utf-8")
    items = load_speechocean(tmp_path)
    assert [i.utt_id for i in items] == ["u1", "u2", "u3"] and items[0].mispronounced == {4}
    heard = {"u1": "W IY K AO R", "u2": "V IY W IH L", "u3": "P L IY Z K AO L"}
    res = speechocean_eval(items, lambda p: Recognition(tuple(heard[p.stem].split())))
    assert res["phone_detection"]["tp"] == 2 and res["phone_detection"]["fp"] == 0
    assert res["utterances"] == 3


def test_audio_is_not_stored_by_default_and_purge_works(tmp_path):
    """Problem 10: no recording is kept unless the user turns it on, and old files are deleted."""
    x = np.zeros(1600, dtype=np.float32)
    assert store(x, 16_000, tmp_path, enabled=False, days=7) is None and not list(tmp_path.iterdir())
    p = store(x, 16_000, tmp_path, enabled=True, days=7)
    assert p.exists() and p.name.startswith("rec-")
    old = time.time() - 10 * 86400
    os.utime(p, (old, old))
    assert purge(tmp_path, 7) == 1 and not p.exists()


def test_settings():
    s = Settings.from_env({}, dotenv=None)
    assert not s.store_audio and s.retention_days == 7 and s.llm_base_url is None
    with pytest.raises(ConfigError):
        Settings.from_env({"SPEAKRIGHT_STORE_AUDIO": "maybe"}, dotenv=None)
    with pytest.raises(ConfigError):
        Settings.from_env({"SPEAKRIGHT_RETENTION_DAYS": "-1"}, dotenv=None)


def test_recognizer_loads_once(monkeypatch):
    """Problem 8: the model loads once for each process, not on each request."""
    made = []

    class Fake:
        def __init__(self, name, device):
            made.append(name)

    monkeypatch.setattr(rec_mod, "CtcPhoneRecognizer", Fake)
    rec_mod.get_recognizer.cache_clear()
    a = rec_mod.get_recognizer("m", "cpu")
    b = rec_mod.get_recognizer("m", "cpu")
    assert a is b and made == ["m"]
    rec_mod.get_recognizer.cache_clear()


def test_cli(tmp_path, capsys):
    assert main(["g2p", "this ship"]) == 0
    assert json.loads(capsys.readouterr().out)[1]["phones"] == "SH IH P"
    assert main(["g2p", "你好", "--lang", "zh"]) == 0
    assert json.loads(capsys.readouterr().out)["with_sandhi"] == ["ni2", "hao3"]
    assert main(["assess", "--text", "this ship", "--phones", "D IH S S IH P"]) == 0
    out = capsys.readouterr().out
    assert "th-stopping" in out and "sh-s-confusion" in out
    wav = write_wav(tmp_path / "t.wav", synth_tones([3, 3], base_hz=170.0))
    assert main(["assess", "--text", "你好", "--lang", "zh", "--phones", "ni hao", "--audio", str(wav), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"][0]["kind"] == "tone-2-as-3"
    assert main(["synth-tones", "--pinyin", "ma1 ma2 ma3 ma4", "--out", str(tmp_path / "m.wav")]) == 0
    capsys.readouterr()
    assert main(["tones", "--audio", str(tmp_path / "m.wav"), "--syllables", "4"]) == 0
    assert [t["tone"] for t in json.loads(capsys.readouterr().out)["tones"]] == [1, 2, 3, 4]
    assert main(["samples", "--lang", "zh"]) == 0
    assert main(["assess", "--text", "x", "--lang", "zh", "--audio", str(wav)]) == 2
    assert main(["assess", "--text", "this", "--phones", "QQ"]) == 2
    assert main(["assess", "--text", "this"]) == 2


def test_cli_demo(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "demo_summary.json").read_text(encoding="utf-8"))
    assert s["eval_english"]["word_detection"]["recall"] > s["eval_english"]["legacy_word_detection"]["recall"]


def test_streamlit_app_runs():
    """Problem 7: browser recording UI. It renders without a recogniser."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import speakright
    app = os.path.join(os.path.dirname(speakright.__file__), "ui", "streamlit_app.py")
    at = AppTest.from_file(app, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "speakright"
