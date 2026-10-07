"""Audio input: WAV read and write (stdlib `wave`), mono, resample to 16 kHz, energy VAD and checks."""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_SR = 16_000


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class AudioCheck:
    duration_s: float
    peak: float
    rms_db: float
    clipped_fraction: float
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def read_wav(src: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Return float32 mono samples in [-1, 1] and the sample rate. PCM 8, 16 or 32 bit."""
    try:
        fh = wave.open(io.BytesIO(src) if isinstance(src, bytes) else str(src), "rb")
    except (wave.Error, EOFError, FileNotFoundError) as exc:
        raise AudioError(f"cannot read WAV: {exc}") from exc
    with fh:
        sr, ch, width, n = fh.getframerate(), fh.getnchannels(), fh.getsampwidth(), fh.getnframes()
        raw = fh.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioError(f"unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def write_wav(path: str | Path, x: np.ndarray, sr: int = TARGET_SR) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    return path


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x.astype(np.float32)
    g = np.gcd(sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)


def frame_energy_db(x: np.ndarray, sr: int, frame_s: float = 0.02) -> np.ndarray:
    n = max(1, int(sr * frame_s))
    k = len(x) // n
    if k == 0:
        return np.array([-120.0])
    frames = x[: k * n].reshape(k, n)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def vad_trim(x: np.ndarray, sr: int, threshold_db: float = 30.0, pad_s: float = 0.1) -> np.ndarray:
    """Remove leading and trailing frames more than `threshold_db` below the loudest frame."""
    e = frame_energy_db(x, sr)
    voiced = np.nonzero(e > e.max() - threshold_db)[0]
    if len(voiced) == 0:
        return x[:0]
    n = int(sr * 0.02)
    pad = int(sr * pad_s)
    return x[max(0, voiced[0] * n - pad): min(len(x), (voiced[-1] + 1) * n + pad)]


def check(x: np.ndarray, sr: int, min_s: float = 0.3, max_s: float = 30.0) -> AudioCheck:
    dur = len(x) / sr if sr else 0.0
    peak = float(np.abs(x).max()) if len(x) else 0.0
    rms_db = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)) if len(x) else -120.0
    clipped = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    problems = []
    if dur < min_s:
        problems.append(f"too short ({dur:.2f} s, minimum {min_s} s)")
    if dur > max_s:
        problems.append(f"too long ({dur:.1f} s, maximum {max_s} s)")
    if rms_db < -45:
        problems.append(f"too quiet ({rms_db:.0f} dBFS)")
    if clipped > 0.01:
        problems.append(f"clipped ({clipped:.1%} of samples)")
    return AudioCheck(round(dur, 3), round(peak, 4), round(rms_db, 1), round(clipped, 4), tuple(problems))


def prepare(src: str | Path | bytes) -> tuple[np.ndarray, AudioCheck]:
    """Read, resample to 16 kHz, trim silence and check. Raise AudioError if the check fails."""
    x, sr = read_wav(src)
    x = vad_trim(to_16k(x, sr), TARGET_SR)
    c = check(x, TARGET_SR)
    if not c.ok:
        raise AudioError("; ".join(c.problems))
    return x, c
