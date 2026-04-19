"""Visual + audio utilities for the debugger UI.

- `build_waveform_chart` produces an Altair layered chart: a downsampled
  waveform under a per-word confidence heatmap, so you can literally *see*
  where Picovoice got unsure.
- `word_slice_wav_bytes` turns a single Word's time range into an RFC 1867
  compliant 16-bit PCM WAV byte stream so `st.audio` can play it.
- `HOMOPHONES` and `flag_homophones` tag known-ambiguous tokens Picovoice
  regularly confuses (homophones, digit words, contractions).
- `transcript_stability` computes which tokens survive across N takes.

Kept isolated from the main app so the UI code stays readable.
"""

from __future__ import annotations

import io
import wave
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import AUDIO
from transcriber import Word


# ---------------------------------------------------------------------------
# Waveform + confidence overlay (Altair, no extra deps)
# ---------------------------------------------------------------------------


def _downsample_waveform(audio: np.ndarray, target: int = 600) -> np.ndarray:
    """Envelope-downsample to `target` points. Preserves visual peaks."""
    if audio.ndim > 1:
        audio = audio.flatten()
    if audio.size == 0:
        return np.zeros(target, dtype=np.float32)
    float_audio = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
    if float_audio.size <= target:
        # Pad with zeros to keep a predictable chart width.
        out = np.zeros(target, dtype=np.float32)
        out[: float_audio.size] = float_audio
        return out
    # Take the max-abs per window so visual peaks survive.
    window = float_audio.size // target
    trimmed = float_audio[: window * target]
    reshaped = trimmed.reshape(target, window)
    env = np.max(np.abs(reshaped), axis=1) * np.sign(reshaped[:, 0])
    return env.astype(np.float32)


def build_waveform_chart(
    audio: np.ndarray,
    words: Sequence[Word],
    sample_rate: int = AUDIO.sample_rate,
    thresholds: Tuple[float, float] = (0.60, 0.85),
):
    """Return an Altair chart: waveform line + colored confidence bands.

    Color bands span each word's [start_sec, end_sec] using the same
    red/amber/green scheme as the transcript so the eye can instantly
    match dips to audio regions.
    """
    import altair as alt  # streamlit bundles altair - no new dep
    import pandas as pd

    low, high = thresholds

    # Waveform dataframe.
    preview = _downsample_waveform(audio, target=800)
    duration = audio.size / sample_rate if audio.size else len(preview) / sample_rate
    xs = np.linspace(0, duration, num=preview.size, endpoint=False)
    wave_df = pd.DataFrame({"t": xs, "amplitude": preview})

    waveform = (
        alt.Chart(wave_df)
        .mark_area(opacity=0.6, color="#7a7a7a")
        .encode(
            x=alt.X("t:Q", title="time (s)"),
            y=alt.Y(
                "amplitude:Q",
                title="amplitude",
                scale=alt.Scale(domain=[-1, 1]),
            ),
        )
    )

    # Overlay bands per word.
    band_rows = []
    for w in words:
        if w.end_sec <= w.start_sec:
            continue
        if w.confidence >= high:
            color = "high"
        elif w.confidence >= low:
            color = "mid"
        else:
            color = "low"
        band_rows.append(
            {
                "word": w.word,
                "start": w.start_sec,
                "end": w.end_sec,
                "confidence": w.confidence,
                "bucket": color,
            }
        )

    if band_rows:
        band_df = pd.DataFrame(band_rows)
        overlay = (
            alt.Chart(band_df)
            .mark_rect(opacity=0.25)
            .encode(
                x="start:Q",
                x2="end:Q",
                y=alt.value(0),
                y2=alt.value(300),
                color=alt.Color(
                    "bucket:N",
                    scale=alt.Scale(
                        domain=["high", "mid", "low"],
                        range=["#1b873a", "#b6791b", "#b3261e"],
                    ),
                    legend=alt.Legend(title="confidence"),
                ),
                tooltip=[
                    alt.Tooltip("word:N"),
                    alt.Tooltip("confidence:Q", format=".2f"),
                    alt.Tooltip("start:Q", format=".2f", title="start (s)"),
                    alt.Tooltip("end:Q", format=".2f", title="end (s)"),
                ],
            )
        )
        chart = (overlay + waveform).properties(height=260)
    else:
        chart = waveform.properties(height=260)

    return chart.configure_view(stroke=None)


# ---------------------------------------------------------------------------
# Per-word audio slicing for st.audio
# ---------------------------------------------------------------------------


def word_slice_wav_bytes(
    audio: np.ndarray,
    word: Word,
    sample_rate: int = AUDIO.sample_rate,
    pad_ms: float = 80.0,
) -> Optional[bytes]:
    """Return a 16-bit PCM WAV byte stream containing just this word's audio."""
    if audio.ndim > 1:
        audio = audio.flatten()
    if audio.size == 0 or word.end_sec <= word.start_sec:
        return None

    pad = int((pad_ms / 1000.0) * sample_rate)
    start = max(0, int(word.start_sec * sample_rate) - pad)
    end = min(audio.size, int(word.end_sec * sample_rate) + pad)
    if end <= start:
        return None

    slice_ = audio[start:end]
    if slice_.dtype != np.int16:
        slice_ = (slice_.astype(np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(slice_.tobytes())
    return buf.getvalue()


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = AUDIO.sample_rate,
) -> bytes:
    """Encode the full audio buffer as a WAV byte stream."""
    if audio.ndim > 1:
        audio = audio.flatten()
    if audio.dtype != np.int16:
        audio = (audio.astype(np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Homophone / ambiguity flagger
# ---------------------------------------------------------------------------


# Known-fragile tokens grouped by acoustic ambiguity class. These are the
# words that trip Picovoice (and every other word-level ASR) most often.
HOMOPHONE_GROUPS: List[List[str]] = [
    ["to", "too", "two"],
    ["there", "their", "theyre", "they're"],
    ["your", "youre", "you're"],
    ["its", "it's"],
    ["for", "four", "fore"],
    ["ate", "eight"],
    ["one", "won"],
    ["sea", "see"],
    ["write", "right", "rite"],
    ["hear", "here"],
    ["knew", "new"],
    ["know", "no"],
    ["buy", "by", "bye"],
    ["our", "hour", "are"],
    ["meet", "meat"],
    ["weak", "week"],
    ["mail", "male"],
    ["son", "sun"],
    ["flower", "flour"],
    ["principal", "principle"],
    ["accept", "except"],
    ["affect", "effect"],
    ["brake", "break"],
    ["cell", "sell"],
    ["whole", "hole"],
    ["red", "read"],
    ["peace", "piece"],
    ["role", "roll"],
    ["cite", "site", "sight"],
    ["waist", "waste"],
]

# Digit-word confusions: Picovoice commonly swaps these with numerals or
# each other.
DIGIT_WORDS = {
    "zero", "oh", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand",
}


def _normalize(token: str) -> str:
    return "".join(c.lower() for c in token if c.isalnum() or c == "'")


HOMOPHONE_LOOKUP = {}
for group in HOMOPHONE_GROUPS:
    label = "/".join(group)
    for w in group:
        HOMOPHONE_LOOKUP[_normalize(w)] = label


def homophone_tag(word: str) -> Optional[str]:
    """Return a group label (e.g. 'to/too/two') if `word` is ambiguous."""
    norm = _normalize(word)
    if not norm:
        return None
    if norm in HOMOPHONE_LOOKUP:
        return HOMOPHONE_LOOKUP[norm]
    if norm in DIGIT_WORDS:
        return "number-word"
    if any(c.isdigit() for c in norm):
        return "numeric"
    return None


# ---------------------------------------------------------------------------
# Transcript stability (Consistency Lab)
# ---------------------------------------------------------------------------


def transcript_stability(
    transcripts: Sequence[str],
) -> dict:
    """Compute per-token stability across N takes.

    Returns {
        'tokens': [{'token': str, 'count': int, 'takes': [idx, ...]}, ...]
                  sorted by -count (most stable first),
        'take_count': N,
        'stable_ratio': fraction of unique tokens seen in every take,
        'token_variance': stddev of transcript length across takes,
    }
    """
    if not transcripts:
        return {
            "tokens": [],
            "take_count": 0,
            "stable_ratio": 0.0,
            "token_variance": 0.0,
        }

    per_take = [
        [_normalize(t) for t in trans.split() if _normalize(t)]
        for trans in transcripts
    ]

    # For each unique token, which takes contain it?
    token_takes: dict[str, List[int]] = {}
    for i, toks in enumerate(per_take):
        for tok in set(toks):
            token_takes.setdefault(tok, []).append(i)

    tokens_rows = [
        {"token": tok, "count": len(takes), "takes": sorted(takes)}
        for tok, takes in token_takes.items()
    ]
    tokens_rows.sort(key=lambda r: (-r["count"], r["token"]))

    n = len(per_take)
    stable = sum(1 for row in tokens_rows if row["count"] == n)
    lengths = [len(toks) for toks in per_take]

    return {
        "tokens": tokens_rows,
        "take_count": n,
        "stable_ratio": stable / max(1, len(tokens_rows)),
        "token_variance": float(np.std(lengths)) if lengths else 0.0,
    }
