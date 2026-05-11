"""Optional offline Vosk partial hints (CREATE-TKS-style feedback).

Mirrors the spirit of ``inputs/voice.py`` in CREATE-TKS: feed 16 kHz mono
int16 audio in chunks, surface the last partial hypothesis before the final
utterance lands. Requires ``pip install vosk`` and a downloaded model;
path comes from ``VOSK_MODEL_PATH`` in the environment or ``.env``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VoskHintResult:
    text: str
    partial_last: str
    available: bool
    detail: str = ""


def vosk_offline_hints(
    audio_int16: Any,
    model_path: str | Path | None,
    *,
    chunk_samples: int = 4000,
) -> VoskHintResult:
    """Run a lightweight streaming pass over ``audio_int16`` (1-D int16)."""
    import numpy as np

    if model_path is None:
        return VoskHintResult("", "", False, "no model path")

    path = Path(model_path).expanduser()
    if not path.is_dir():
        return VoskHintResult("", "", False, f"model path missing: {path}")

    try:
        from vosk import KaldiRecognizer, Model
    except Exception as exc:
        return VoskHintResult("", "", False, f"vosk import failed: {exc}")

    audio = np.asarray(audio_int16, dtype=np.int16).flatten()
    if audio.size == 0:
        return VoskHintResult("", "", True, "empty audio")

    try:
        model = Model(str(path))
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
    except Exception as exc:
        return VoskHintResult("", "", False, f"vosk init failed: {exc}")

    partial_last = ""
    step = max(512, int(chunk_samples))
    try:
        for start in range(0, audio.size, step):
            chunk = audio[start : start + step].tobytes()
            if rec.AcceptWaveform(chunk):
                try:
                    partial_last = ""
                except Exception:
                    pass
            else:
                try:
                    pr = json.loads(rec.PartialResult()).get("partial", "")
                    pr = (pr or "").strip()
                    if pr:
                        partial_last = pr
                except Exception:
                    pass

        tail = rec.FinalResult()
        try:
            final_j = json.loads(tail)
        except Exception:
            final_j = {}
        text = str(final_j.get("text", "")).strip()
    finally:
        try:
            del model
        except Exception:
            pass

    return VoskHintResult(
        text=text,
        partial_last=partial_last.strip(),
        available=True,
        detail="ok",
    )
