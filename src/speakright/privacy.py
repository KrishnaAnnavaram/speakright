"""Voice recordings are personal data. By default speakright keeps no recording.

With `SPEAKRIGHT_STORE_AUDIO=yes`, `store` writes the recording with a random name (no user name),
and `purge` deletes recordings older than the retention period. `store` calls `purge` first.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from .audio import write_wav


def purge(audio_dir: str | Path, days: int, now: float | None = None) -> int:
    d = Path(audio_dir)
    if not d.is_dir():
        return 0
    limit = (now if now is not None else time.time()) - days * 86400
    removed = 0
    for f in d.glob("rec-*.wav"):
        if f.stat().st_mtime < limit:
            f.unlink()
            removed += 1
    return removed


def store(x: np.ndarray, sr: int, audio_dir: str | Path, enabled: bool, days: int) -> Path | None:
    if not enabled:
        return None
    purge(audio_dir, days)
    return write_wav(Path(audio_dir) / f"rec-{uuid.uuid4().hex}.wav", x, sr)
