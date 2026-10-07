# Data

No audio, no model and no dataset is in this repository. Voice recordings are personal
(biometric) data. Git ignores all audio file types.

## Evaluation corpora

| Corpus | Source | Licence / terms | Use in speakright |
|---|---|---|---|
| speechocean762 | OpenSLR 101, https://www.openslr.org/101/ | CC BY 4.0 | `speakright eval --speechocean <folder>`: phone-error detection and correlation with the human `total` score |
| L2-ARCTIC | https://psi.engr.tamu.edu/l2-arctic-corpus/ | Research licence (read it before use) | Not read yet. Planned loader |
| AISHELL-1 | OpenSLR 33, https://www.openslr.org/33/ | Apache 2.0 | Not read yet. Planned for Mandarin tone checks |

### speechocean762 layout that the loader expects

```
<folder>/
  resource/scores.json      # {utt_id: {"text", "total", "words": [{"phones", "mispronunciations": [{"index"}]}]}}
  test/wav.scp              # "<utt_id> <relative path to the WAV>"
  WAVE/...                  # the WAV files
```

`mispronunciations[].index` is the 0-based position of the phone inside its word. The loader
changes it to a position in the utterance.

## Pronunciation sources

| Source | How to use it |
|---|---|
| Built-in English lexicon | Covers all built-in practice sentences. No action |
| CMUdict | `pip install -e ".[g2p]"` (package `cmudict`), or a CMUdict text file in `SPEAKRIGHT_CMUDICT` |
| Built-in pinyin table | Covers all built-in Mandarin sentences. No action |
| pypinyin | `pip install -e ".[g2p]"` for any Chinese text |

## Recordings

By default no recording is written. With `SPEAKRIGHT_STORE_AUDIO=yes`, recordings go to
`SPEAKRIGHT_AUDIO_DIR` with random names, and recordings older than `SPEAKRIGHT_RETENTION_DAYS`
are deleted at the next store, or with `speakright purge`.

## Synthetic data

The tests and `speakright demo` use simulated learners (`simulate.py`) and hummed tone audio
(`pitch.synth_tones`). They need no download and no network.
