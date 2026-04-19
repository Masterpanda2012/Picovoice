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


def get_access_key() -> str | None:
    """Return the Picovoice AccessKey from env, or None if not configured."""
    key = os.environ.get("PICOVOICE_ACCESS_KEY")
    if key:
        return key.strip()
    return None


def get_elevenlabs_api_key() -> str | None:
    """Return the ElevenLabs API key from env, or None if not configured.

    Treats the `.env.example` placeholder as "unset" so users who forgot to
    edit their .env don't get confusing 401s downstream.
    """
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return None
    key = key.strip()
    if not key or key == "your-elevenlabs-api-key-here":
        return None
    return key
