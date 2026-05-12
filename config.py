"""Central configuration for the Voice AI Debugger.

Loads the Picovoice AccessKey from the environment (or a local .env file) so
the key is never committed to source control. All Picovoice engines
(Leopard, Cheetah, Cobra) expect 16 kHz mono 16-bit PCM audio, so those
constants live here too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Tiny .env loader so we don't add python-dotenv as a hard dependency."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    default_duration_seconds: float = 5.0


@dataclass(frozen=True)
class ConfidenceConfig:
    # Words at/above `high` render normally (green).
    high: float = 0.85
    # Words at/above `low` but below `high` render amber.
    low: float = 0.60
    # Default threshold for the overall-confidence warning banner.
    default_threshold: float = 0.75


AUDIO = AudioConfig()
CONFIDENCE = ConfidenceConfig()


def _is_picovoice_placeholder(key: str) -> bool:
    k = (key or "").strip().lower()
    return not k or k == "your-picovoice-access-key-here" or k.startswith("your-picovoice")


def _is_elevenlabs_placeholder(key: str) -> bool:
    k = (key or "").strip().lower()
    return not k or k == "your-elevenlabs-api-key-here" or k.startswith("your-elevenlabs")


def normalize_session_picovoice_key(raw: str | None) -> str | None:
    """Validate a user-pasted Picovoice key (session UI). None if empty/placeholder."""
    if raw is None:
        return None
    s = raw.strip()
    if _is_picovoice_placeholder(s):
        return None
    return s


def normalize_session_elevenlabs_key(raw: str | None) -> str | None:
    """Validate a user-pasted ElevenLabs key (session UI). None if empty/placeholder."""
    if raw is None:
        return None
    s = raw.strip()
    if _is_elevenlabs_placeholder(s):
        return None
    return s


def get_access_key() -> str | None:
    """Return the Picovoice AccessKey from env, or None if not configured.

    Treats the `.env.example` placeholder as "unset" so Picovoice SDKs are
    never fed a fake key (which produces opaque init errors).
    """
    key = os.environ.get("PICOVOICE_ACCESS_KEY")
    if not key:
        return None
    key = key.strip()
    if _is_picovoice_placeholder(key):
        return None
    return key


def get_elevenlabs_api_key() -> str | None:
    """Return the ElevenLabs API key from env, or None if not configured.

    Treats the `.env.example` placeholder as "unset" so users who forgot to
    edit their .env don't get confusing 401s downstream.
    """
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return None
    key = key.strip()
    if _is_elevenlabs_placeholder(key):
        return None
    return key


def get_vosk_model_path() -> Path | None:
    """Directory containing a Vosk model (see https://alphacephei.com/vosk/models)."""
    raw = os.environ.get("VOSK_MODEL_PATH")
    if not raw:
        return None
    p = Path(raw.strip()).expanduser()
    if not p.is_dir():
        return None
    return p
