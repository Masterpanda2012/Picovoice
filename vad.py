"""Optional Cobra VAD integration.

Cobra consumes fixed-size frames of 16 kHz mono int16 audio and returns a
voice probability per frame. We use it to drop silent regions before
sending audio to Leopard - the architecturally correct pipeline Picovoice
recommends for production.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config import AUDIO


class VADError(RuntimeError):
    pass


class CobraVAD:
    def __init__(self, access_key: str, threshold: float = 0.5):
        if not access_key:
            raise VADError("Missing Picovoice AccessKey.")
        try:
            import pvcobra
        except Exception as exc:
            raise VADError(
                "pvcobra is not installed. Run `pip install pvcobra`."
            ) from exc

        try:
            self._cobra = pvcobra.create(access_key=access_key)
        except Exception as exc:
            raise VADError(f"Failed to initialise Cobra: {exc}") from exc

        self.frame_length: int = int(self._cobra.frame_length)
        self.sample_rate: int = int(self._cobra.sample_rate)
        self.threshold = float(threshold)

    def voice_probabilities(self, audio: np.ndarray) -> List[float]:
        """Run Cobra frame-by-frame across `audio` and return a probability list."""
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        audio = audio.flatten()

        probs: List[float] = []
        for start in range(0, len(audio) - self.frame_length + 1, self.frame_length):
            frame = audio[start : start + self.frame_length]
            try:
                probs.append(float(self._cobra.process(frame)))
            except Exception as exc:
                raise VADError(f"Cobra processing failed: {exc}") from exc
        return probs

    def filter_voiced(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (voiced_audio, voiced_ratio) keeping only frames above threshold."""
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        audio = audio.flatten()

        kept_frames: List[np.ndarray] = []
        total = 0
        voiced = 0
        for start in range(0, len(audio) - self.frame_length + 1, self.frame_length):
            frame = audio[start : start + self.frame_length]
            total += 1
            try:
                p = float(self._cobra.process(frame))
            except Exception as exc:
                raise VADError(f"Cobra processing failed: {exc}") from exc
            if p >= self.threshold:
                voiced += 1
                kept_frames.append(frame)

        if not kept_frames:
            return np.zeros(0, dtype=np.int16), 0.0

        ratio = voiced / total if total else 0.0
        return np.concatenate(kept_frames).astype(np.int16), ratio

    def delete(self) -> None:
        try:
            self._cobra.delete()
        except Exception:
            pass
