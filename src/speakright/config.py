"""Settings from environment variables and an optional local .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    pass


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    p = Path(path)
    out: dict[str, str] = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip().strip("'\"")
    return out


@dataclass(frozen=True)
class Settings:
    recognizer_model: str
    device: str
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    store_audio: bool
    audio_dir: Path
    retention_days: int
    cmudict: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: str | Path | None = ".env") -> "Settings":
        e: dict[str, str] = {}
        if dotenv is not None:
            e.update(load_dotenv(dotenv))
        e.update(os.environ if env is None else env)
        try:
            days = int(e.get("SPEAKRIGHT_RETENTION_DAYS") or 7)
        except ValueError as exc:
            raise ConfigError("SPEAKRIGHT_RETENTION_DAYS must be an integer") from exc
        if days < 0:
            raise ConfigError("SPEAKRIGHT_RETENTION_DAYS must be >= 0")
        store = (e.get("SPEAKRIGHT_STORE_AUDIO") or "no").lower()
        if store not in ("yes", "no"):
            raise ConfigError("SPEAKRIGHT_STORE_AUDIO must be yes or no")
        return cls(
            recognizer_model=e.get("SPEAKRIGHT_RECOGNIZER_MODEL") or "facebook/wav2vec2-lv-60-espeak-cv-ft",
            device=e.get("SPEAKRIGHT_DEVICE") or "cpu",
            llm_base_url=e.get("SPEAKRIGHT_LLM_BASE_URL") or None,
            llm_api_key=e.get("SPEAKRIGHT_LLM_API_KEY") or None,
            llm_model=e.get("SPEAKRIGHT_LLM_MODEL") or None,
            store_audio=store == "yes",
            audio_dir=Path(e.get("SPEAKRIGHT_AUDIO_DIR") or "artifacts/audio"),
            retention_days=days,
            cmudict=e.get("SPEAKRIGHT_CMUDICT") or None,
        )
