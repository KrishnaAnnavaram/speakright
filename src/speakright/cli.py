"""The `speakright` command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assess import AssessError, Assessor, to_markdown
from .audio import AudioError, prepare, write_wav
from .config import Settings
from .g2p_en import EnglishG2P, G2PError
from .llm import build_llm
from .phones import PhoneError, parse_phones
from .pinyin import PinyinError, apply_sandhi, parse_pinyin, target_syllables
from .privacy import purge, store
from .recognize import Recognition


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _assessor(s: Settings) -> Assessor:
    return Assessor(llm=build_llm(s.llm_base_url, s.llm_api_key, s.llm_model), g2p=EnglishG2P(s.cmudict))


def cmd_g2p(args, s):
    if args.lang == "en":
        _print([{"word": w.word, "phones": " ".join(w.phones), "source": w.source, "approximate": w.approximate}
                for w in EnglishG2P(s.cmudict).sentence(args.text)])
    else:
        raw = target_syllables(args.text)
        chars = "".join(c for c in args.text if "一" <= c <= "鿿") or None
        _print({"tone3": raw, "with_sandhi": apply_sandhi(raw, chars)})


def cmd_assess(args, s):
    audio = None
    if args.phones:
        units = parse_phones(args.phones) if args.lang == "en" else parse_pinyin(args.phones)
        rec = Recognition(tuple(units), source="manual")
        if args.audio:
            audio, _ = prepare(args.audio)  # Mandarin: the tones come from the audio
    elif args.audio:
        if args.lang == "zh":
            raise AssessError("Mandarin needs --phones with the heard syllables; the tones come from --audio")
        from .recognize import get_recognizer

        audio, _ = prepare(args.audio)
        rec = get_recognizer(s.recognizer_model, s.device).recognize(audio, 16_000)
    else:
        raise AssessError("give --audio, --phones or both")
    result = _assessor(s).assess(args.lang, args.text, rec, audio)
    if audio is not None:
        store(audio, 16_000, s.audio_dir, s.store_audio, s.retention_days)
    print(result.to_json() if args.json else to_markdown(result))


def cmd_tones(args, s):
    from .pitch import tones_from_audio

    audio, check = prepare(args.audio)
    res = tones_from_audio(audio, 16_000, args.syllables)
    _print({"audio": check.__dict__, "tones": [r.__dict__ for r in res]})


def cmd_synth_tones(args, s):
    from .pitch import synth_tones

    tones = [int(x[-1]) for x in parse_pinyin(args.pinyin)]
    path = write_wav(args.out, synth_tones(tones, base_hz=args.base_hz))
    _print({"out": str(path), "tones": tones})


def cmd_eval(args, s):
    from .evaluate import synthetic_english, synthetic_mandarin

    if args.speechocean:
        from .evaluate import load_speechocean, speechocean_eval
        from .recognize import get_recognizer

        rec = get_recognizer(s.recognizer_model, s.device)
        items = load_speechocean(args.speechocean)[: args.limit]
        _print(speechocean_eval(items, lambda p: rec.recognize(prepare(p)[0], 16_000)))
        return
    _print({"english": synthetic_english(args.n, seed=args.seed),
            "mandarin": synthetic_mandarin(max(10, args.n // 2), seed=args.seed),
            "mandarin_flat_speech": synthetic_mandarin(max(10, args.n // 2), seed=args.seed, noise=0.05,
                                                       range_scale=0.5)})


def cmd_demo(args, s):
    from .evaluate import synthetic_english, synthetic_mandarin
    from .pitch import synth_tones

    a = Assessor()
    en = a.assess("en", "three thin thieves thought a thousand thoughts",
                  Recognition(tuple("T R IY T IH N TH IY V Z S AO T AH T AW Z AH N D TH AO T S".split()),
                              tuple([0.9] * 24)))
    audio = synth_tones([2, 3], base_hz=180.0)   # the learner says ni2 hao3 with the right sandhi
    zh = a.assess("zh", "你好", Recognition(("ni5", "hao5")), audio)
    audio2 = synth_tones([3, 3], base_hz=180.0)  # no sandhi: tone 3 + tone 3
    zh2 = a.assess("zh", "你好", Recognition(("ni5", "hao5")), audio2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"english_example": en.to_dict(), "mandarin_example_correct": zh.to_dict(),
               "mandarin_example_no_sandhi": zh2.to_dict(),
               "eval_english": synthetic_english(200, seed=0),
               "eval_mandarin": synthetic_mandarin(100, seed=0),
               "eval_mandarin_flat_speech": synthetic_mandarin(100, seed=0, noise=0.05, range_scale=0.5)}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    print(to_markdown(en))
    print()
    print(to_markdown(zh2))
    print()
    _print({k: v for k, v in summary.items() if k.startswith("eval")})


def cmd_samples(args, s):
    from .samples import samples

    _print(samples(args.lang))


def cmd_purge(args, s):
    _print({"removed": purge(s.audio_dir, args.days if args.days is not None else s.retention_days)})


def cmd_ui(args, s):
    import subprocess

    app = Path(__file__).parent / "ui" / "streamlit_app.py"
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *args.rest]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speakright", description="Phone-level pronunciation assessment.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("g2p", help="canonical phones (en) or TONE3 pinyin with sandhi (zh)")
    c.add_argument("text")
    c.add_argument("--lang", choices=["en", "zh"], default="en")
    c.set_defaults(func=cmd_g2p)

    c = sub.add_parser("assess", help="assess one utterance")
    c.add_argument("--text", required=True)
    c.add_argument("--lang", choices=["en", "zh"], default="en")
    c.add_argument("--audio", type=Path, help="WAV file")
    c.add_argument("--phones", help="heard units: ARPAbet (en) or pinyin syllables (zh)")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_assess)

    c = sub.add_parser("tones", help="Mandarin tones from a WAV file (pitch only)")
    c.add_argument("--audio", type=Path, required=True)
    c.add_argument("--syllables", type=int, required=True)
    c.set_defaults(func=cmd_tones)

    c = sub.add_parser("synth-tones", help="write a hummed WAV with the tones of a pinyin string")
    c.add_argument("--pinyin", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--base-hz", type=float, default=170.0)
    c.set_defaults(func=cmd_synth_tones)

    c = sub.add_parser("eval", help="synthetic evaluation, or speechocean762 with a real recogniser")
    c.add_argument("--n", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--speechocean", type=Path)
    c.add_argument("--limit", type=int, default=500)
    c.set_defaults(func=cmd_eval)

    c = sub.add_parser("demo", help="offline examples and the synthetic evaluation")
    c.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    c.set_defaults(func=cmd_demo)

    c = sub.add_parser("samples", help="built-in practice sentences")
    c.add_argument("--lang", choices=["en", "zh"], default="en")
    c.set_defaults(func=cmd_samples)

    c = sub.add_parser("purge", help="delete stored recordings older than the retention period")
    c.add_argument("--days", type=int)
    c.set_defaults(func=cmd_purge)

    c = sub.add_parser("ui", help="browser UI (extra ui)")
    c.add_argument("rest", nargs=argparse.REMAINDER)
    c.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Chinese text on a console with a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        args.func(args, Settings.from_env())
    except (AssessError, AudioError, G2PError, PinyinError, PhoneError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
