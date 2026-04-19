"""Microphone capture utilities.

Uses `sounddevice` to capture mono 16 kHz 16-bit PCM audio, which is the
format every Picovoice engine expects. Returns a flat numpy int16 array
so it can be passed straight into Leopard / Cheetah / Cobra.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import AUDIO


class RecorderError(RuntimeError):
    pass


def record_fixed_duration(duration_seconds: float) -> np.ndarray:
    """Block the caller, record `duration_seconds` of mic audio, return int16 PCM."""
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on host audio stack
        raise RecorderError(
            "sounddevice is not available. Install it with `pip install sounddevice` "
            "and ensure PortAudio is installed on your system."
        ) from exc

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    frames = int(duration_seconds * AUDIO.sample_rate)
    try:
        audio = sd.rec(
            frames,
            samplerate=AUDIO.sample_rate,
            channels=AUDIO.channels,
            dtype=AUDIO.dtype,
        )
        sd.wait()
    except Exception as exc:
        raise RecorderError(f"Failed to record audio: {exc}") from exc

    return np.asarray(audio, dtype=np.int16).flatten()


def load_wav_file(path: str) -> np.ndarray:
    """Load a 16 kHz mono 16-bit PCM wav file as a flat int16 array.

    If the file is stereo we average channels; if the sample rate differs we
    do a simple linear resample so Leopard still accepts the stream.
    """
    import wave

    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise RecorderError(
            f"Unsupported sample width {sample_width * 8} bit; expected 16-bit PCM."
        )

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if sample_rate != AUDIO.sample_rate:
        # Simple linear resample - fine for dev/demo use.
        ratio = AUDIO.sample_rate / sample_rate
        target_length = int(round(len(audio) * ratio))
        if target_length <= 0:
            raise RecorderError("Audio file is empty after resample.")
        xp = np.linspace(0, 1, num=len(audio), endpoint=False)
        fp = audio.astype(np.float32)
        x = np.linspace(0, 1, num=target_length, endpoint=False)
        audio = np.interp(x, xp, fp).astype(np.int16)

    return audio


def list_input_devices() -> list[dict]:
    """Enumerate available microphones (best-effort, returns [] on failure)."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception:
        return []

    inputs: list[dict] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            inputs.append(
                {
                    "index": idx,
                    "name": dev.get("name", f"device {idx}"),
                    "channels": dev.get("max_input_channels", 0),
                    "sample_rate": dev.get("default_samplerate", 0),
                }
            )
    return inputs
