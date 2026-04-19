"""Audio + transcript diagnostics.

Everything in here is engine-agnostic. Functions here expose the signals
Picovoice itself does not: audio quality, robustness curves, WER against
ground truth, and session-level patterns.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import AUDIO


# ---------------------------------------------------------------------------
# 1. Audio quality pre-flight
# ---------------------------------------------------------------------------


@dataclass
class AudioStats:
    duration_sec: float
    rms_dbfs: float           # full-scale dB of overall energy (<=0, higher = louder)
    peak_dbfs: float          # full-scale dB of the loudest sample
    clipping_ratio: float     # fraction of samples at/near int16 ceiling
    dc_offset: float          # mean sample value normalised to [-1, 1]
    estimated_snr_db: float   # noise-floor SNR estimate (higher = cleaner)
    voiced_fraction: float    # rough VAD via energy gating (0-1)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 3),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "clipping_ratio": round(self.clipping_ratio, 4),
            "dc_offset": round(self.dc_offset, 4),
            "estimated_snr_db": round(self.estimated_snr_db, 2),
            "voiced_fraction": round(self.voiced_fraction, 3),
            "warnings": list(self.warnings),
        }


def _to_float(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def analyze_audio(audio: np.ndarray) -> AudioStats:
    """Compute quality metrics + produce actionable warnings."""
    if audio.ndim > 1:
        audio = audio.flatten()
    if audio.size == 0:
        return AudioStats(0, -120.0, -120.0, 0.0, 0.0, 0.0, 0.0, ["Empty audio."])

    float_audio = _to_float(audio)
    duration_sec = audio.size / AUDIO.sample_rate

    rms = float(np.sqrt(np.mean(float_audio ** 2)) + 1e-12)
    peak = float(np.max(np.abs(float_audio)) + 1e-12)
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-10))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-10))

    # Clipping: how many samples are pinned to the int16 ceiling?
    if audio.dtype == np.int16:
        clipping_ratio = float(np.mean(np.abs(audio) >= 32760))
    else:
        clipping_ratio = float(np.mean(np.abs(float_audio) >= 0.999))

    dc_offset = float(np.mean(float_audio))

    # SNR estimate: compare energy in loud vs quiet windows.
    window = max(1, int(AUDIO.sample_rate * 0.030))  # 30 ms
    trimmed = float_audio[: (float_audio.size // window) * window]
    if trimmed.size >= window:
        frame_rms = np.sqrt(np.mean(trimmed.reshape(-1, window) ** 2, axis=1) + 1e-12)
        frame_db = 20.0 * np.log10(frame_rms + 1e-10)
        # Top 10% = signal, bottom 10% = noise floor
        signal = float(np.percentile(frame_db, 90))
        noise = float(np.percentile(frame_db, 10))
        estimated_snr_db = max(0.0, signal - noise)
        # Voiced fraction: frames within 10 dB of the signal peak.
        voiced_fraction = float(np.mean(frame_db >= (signal - 10.0)))
    else:
        estimated_snr_db = 0.0
        voiced_fraction = 0.0

    warnings: List[str] = []
    if clipping_ratio > 0.005:
        warnings.append(
            f"Clipping detected on {clipping_ratio * 100:.1f}% of samples — "
            "reduce mic gain."
        )
    if rms_dbfs < -40.0:
        warnings.append(
            f"Audio is very quiet (RMS {rms_dbfs:.1f} dBFS). Picovoice may "
            "return low confidence purely from low SNR."
        )
    if rms_dbfs > -6.0:
        warnings.append(
            f"Audio is very hot (RMS {rms_dbfs:.1f} dBFS). Risk of distortion."
        )
    if abs(dc_offset) > 0.02:
        warnings.append(
            f"DC offset {dc_offset:+.3f} detected — check your audio interface "
            "or apply a high-pass filter."
        )
    if estimated_snr_db < 10.0:
        warnings.append(
            f"Estimated SNR is only {estimated_snr_db:.1f} dB. Expect confidence "
            "degradation; consider Cobra VAD + a quieter room."
        )
    if voiced_fraction < 0.15 and duration_sec >= 1.0:
        warnings.append(
            f"Only {voiced_fraction * 100:.0f}% of frames look voiced — the "
            "engine is mostly processing silence."
        )

    return AudioStats(
        duration_sec=duration_sec,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        clipping_ratio=clipping_ratio,
        dc_offset=dc_offset,
        estimated_snr_db=estimated_snr_db,
        voiced_fraction=voiced_fraction,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 2. Noise injection for the Robustness Lab
# ---------------------------------------------------------------------------


def inject_white_noise(
    audio: np.ndarray,
    target_snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a noisy copy of `audio` with approximately `target_snr_db` SNR.

    SNR is computed against the signal's RMS. Output is clipped back to int16
    range so Picovoice accepts it.
    """
    if audio.ndim > 1:
        audio = audio.flatten()
    if audio.size == 0:
        return audio.astype(np.int16)

    rng = rng or np.random.default_rng()
    float_audio = _to_float(audio)

    signal_rms = float(np.sqrt(np.mean(float_audio ** 2)) + 1e-12)
    if signal_rms < 1e-6:
        return audio.astype(np.int16)

    snr_linear = 10.0 ** (target_snr_db / 20.0)
    noise_rms = signal_rms / max(snr_linear, 1e-12)
    noise = rng.standard_normal(float_audio.size).astype(np.float32) * noise_rms

    mixed = float_audio + noise
    # Prevent hard clipping if we went over full-scale.
    peak = float(np.max(np.abs(mixed)) + 1e-12)
    if peak > 0.999:
        mixed = mixed * (0.999 / peak)

    return (mixed * 32767.0).astype(np.int16)


# ---------------------------------------------------------------------------
# 3. Word Error Rate (Ground Truth feedback)
# ---------------------------------------------------------------------------


@dataclass
class WERResult:
    wer: float                          # substitutions+deletions+insertions / ref_len
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    alignment: List[Tuple[str, Optional[str], Optional[str]]]
    # alignment ops: ("match"|"sub"|"del"|"ins", ref_word, hyp_word)


def _normalize_tokens(text: str) -> List[str]:
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return cleaned.split()


def word_error_rate(reference: str, hypothesis: str) -> WERResult:
    """Classic Levenshtein-on-words with aligned edit script."""
    ref = _normalize_tokens(reference)
    hyp = _normalize_tokens(hypothesis)
    R, H = len(ref), len(hyp)

    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # match/sub
            )

    # Backtrace alignment.
    i, j = R, H
    align: List[Tuple[str, Optional[str], Optional[str]]] = []
    subs = dels = ins = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            align.append(("match", ref[i - 1], hyp[j - 1]))
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            align.append(("sub", ref[i - 1], hyp[j - 1]))
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            align.append(("del", ref[i - 1], None))
            dels += 1
            i -= 1
        else:
            align.append(("ins", None, hyp[j - 1]))
            ins += 1
            j -= 1
    align.reverse()

    denom = max(1, R)
    wer = (subs + dels + ins) / denom
    return WERResult(
        wer=wer,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        hits=hits,
        alignment=align,
    )


# ---------------------------------------------------------------------------
# 4. Failure library (persisted to disk for regression testing)
# ---------------------------------------------------------------------------


DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parent / "failure_library.json"
DEFAULT_AUDIO_DIR = Path(__file__).resolve().parent / "failure_library_audio"


def _new_failure_id() -> str:
    # short, sortable-by-time identifier
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")


def append_failure(
    entry: dict,
    audio_int16: Optional[np.ndarray] = None,
    path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
) -> Path:
    """Append `entry` to the failure library, optionally saving the audio too.

    When `audio_int16` is provided we write a 16 kHz mono WAV into
    `audio_dir` (default: ./failure_library_audio/<id>.wav) so later we
    can actually replay the audio through a newer engine.
    """
    path = Path(path) if path else DEFAULT_LIBRARY_PATH
    audio_dir = Path(audio_dir) if audio_dir else DEFAULT_AUDIO_DIR
    existing: List[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    fid = entry.get("id") or _new_failure_id()
    entry = {
        **entry,
        "id": fid,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    if audio_int16 is not None and audio_int16.size > 0:
        import wave

        audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audio_dir / f"{fid}.wav"
        buf = audio_int16
        if buf.dtype != np.int16:
            buf = (buf.astype(np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(AUDIO.sample_rate)
            w.writeframes(buf.tobytes())
        entry["audio_path"] = str(wav_path)

    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2))
    return path


def load_failures(path: Optional[Path] = None) -> List[dict]:
    path = Path(path) if path else DEFAULT_LIBRARY_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_failure_audio(entry: dict) -> Optional[np.ndarray]:
    """Return the stored audio for a failure entry as int16, or None."""
    audio_path = entry.get("audio_path")
    if not audio_path:
        return None
    p = Path(audio_path)
    if not p.exists():
        return None
    import wave

    try:
        with wave.open(str(p), "rb") as w:
            frames = w.readframes(w.getnframes())
            if w.getsampwidth() != 2:
                return None
            return np.frombuffer(frames, dtype=np.int16)
    except Exception:
        return None


def clear_failures(
    path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
) -> None:
    """Delete the library JSON and every associated WAV."""
    path = Path(path) if path else DEFAULT_LIBRARY_PATH
    audio_dir = Path(audio_dir) if audio_dir else DEFAULT_AUDIO_DIR
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
    try:
        if audio_dir.exists():
            for f in audio_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 5. Session insights: confidence by relative position
# ---------------------------------------------------------------------------


def position_confidence(
    utterances: Sequence[Sequence[float]],
    n_buckets: int = 10,
) -> Dict[int, float]:
    """Map relative position bucket (0..n_buckets-1) -> mean confidence.

    For each utterance we resample its per-word confidence sequence into
    `n_buckets` bins, then average across utterances. This proves or
    disproves the "edges are less confident" pattern across a session.
    """
    if n_buckets <= 0:
        return {}

    sums = np.zeros(n_buckets, dtype=np.float64)
    counts = np.zeros(n_buckets, dtype=np.int64)

    for confs in utterances:
        seq = np.asarray(list(confs), dtype=np.float64)
        if seq.size == 0:
            continue
        if seq.size == 1:
            sums[n_buckets // 2] += seq[0]
            counts[n_buckets // 2] += 1
            continue
        # Map each word i (0..N-1) to a bucket in [0, n_buckets).
        xs = np.linspace(0, n_buckets - 1, num=seq.size)
        buckets = np.clip(np.round(xs).astype(int), 0, n_buckets - 1)
        for b, v in zip(buckets, seq):
            sums[b] += v
            counts[b] += 1

    out: Dict[int, float] = {}
    for b in range(n_buckets):
        if counts[b] > 0:
            out[b] = float(sums[b] / counts[b])
    return out
