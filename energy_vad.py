"""RMS energy voice-activity gate — fallback when Picovoice Cobra is unavailable.

CREATE-TKS proves that local on-device feedback (live partials + “heard” line)
builds user trust. Streamlit cannot easily mirror a live Vosk loop, so we
borrow the *silence-stripping* idea from classic energy VAD and expose the
same trust-building *pattern* in the UI: a compact Heard / Partial / backend
panel + optional offline Vosk partial hints when ``VOSK_MODEL_PATH`` is set.

Frame semantics intentionally mirror Cobra's “keep or drop whole frames”
behaviour so downstream STT still receives int16 PCM chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import AUDIO


@dataclass
class EnergyVADReport:
    """Diagnostics for the Debugger tab chart + CREATE-TKS-style captions."""

    noise_floor_dbfs: float
    threshold_dbfs: float
    frame_centers_sec: np.ndarray  # shape (n_frames,)
    frame_dbfs: np.ndarray  # shape (n_frames,)
    voiced_mask: np.ndarray  # bool (n_frames,)


class EnergyGateVAD:
    """Drop quiet audio frames using a noise-adaptive RMS threshold.

    ``threshold`` reuses the Cobra slider semantics on [0.1, 0.9]: higher
    values require a louder signal relative to the estimated noise floor
    (stricter — fewer frames classified as speech).
    """

    def __init__(self, threshold: float = 0.5, *, frame_samples: int | None = None):
        self.threshold = float(threshold)
        fs = frame_samples or max(160, int(AUDIO.sample_rate * 0.03))  # ~30 ms
        self.frame_length: int = fs
        self.sample_rate: int = AUDIO.sample_rate

    @staticmethod
    def _frame_dbfs(frames: np.ndarray) -> np.ndarray:
        """frames: (n, frame_len) int16 -> (n,) dBFS."""
        f = frames.astype(np.float32)
        rms = np.sqrt(np.mean(f * f, axis=1) + 1e-12)
        return 20.0 * np.log10(rms / 32768.0 + 1e-12)

    def _offset_db_for_slider(self) -> float:
        """Map Cobra-like probability threshold to a dB margin above noise."""
        t = max(0.05, min(0.95, self.threshold))
        # 0.1 -> ~4 dB, 0.9 -> ~32 dB (monotonic stricter).
        lo, hi = 0.1, 0.9
        t_norm = (t - lo) / (hi - lo) if hi > lo else 0.5
        t_norm = max(0.0, min(1.0, t_norm))
        return 4.0 + t_norm * 28.0

    def filter_voiced(self, audio: np.ndarray) -> Tuple[np.ndarray, float, EnergyVADReport]:
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        audio = audio.flatten()
        fl = self.frame_length
        if audio.size < fl:
            return audio.copy(), 1.0, EnergyVADReport(
                noise_floor_dbfs=-90.0,
                threshold_dbfs=-90.0,
                frame_centers_sec=np.zeros(0, dtype=np.float64),
                frame_dbfs=np.zeros(0, dtype=np.float64),
                voiced_mask=np.zeros(0, dtype=bool),
            )

        # Whole frames only (same spirit as Cobra loop).
        n_full = (audio.size // fl) * fl
        trimmed = audio[:n_full]
        n_frames = len(trimmed) // fl
        frames = trimmed.reshape(n_frames, fl)

        dbfs = self._frame_dbfs(frames)
        # Robust noise floor: lower quartile of frame energies.
        noise_floor = float(np.percentile(dbfs, 25))
        thresh = noise_floor + self._offset_db_for_slider()
        mask = dbfs >= thresh

        # Bridge short unvoiced gaps (<=120 ms) to avoid chopping words.
        max_gap = max(1, int(round(0.120 * self.sample_rate / fl)))
        mask = _bridge_short_gaps(mask, max_gap)

        centers = (np.arange(n_frames) + 0.5) * fl / self.sample_rate
        report = EnergyVADReport(
            noise_floor_dbfs=noise_floor,
            threshold_dbfs=thresh,
            frame_centers_sec=centers.astype(np.float64),
            frame_dbfs=dbfs.astype(np.float64),
            voiced_mask=mask.astype(bool),
        )

        if not np.any(mask):
            return np.zeros(0, dtype=np.int16), 0.0, report

        kept: List[np.ndarray] = []
        voiced_count = int(mask.sum())
        for i in range(n_frames):
            if mask[i]:
                kept.append(frames[i])
        ratio = voiced_count / n_frames if n_frames else 0.0
        return np.concatenate(kept).astype(np.int16), float(ratio), report

    def delete(self) -> None:
        return None


def _bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False runs shorter than ``max_gap`` frames when sandwiched by True."""
    if mask.size == 0 or max_gap <= 0:
        return mask
    out = mask.copy()
    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        gap_len = j - i
        left_voice = i > 0 and out[i - 1]
        right_voice = j < n and out[j]
        if left_voice and right_voice and gap_len <= max_gap:
            out[i:j] = True
        i = j
    return out
