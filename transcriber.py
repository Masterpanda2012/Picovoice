"""Thin wrappers around Picovoice Leopard (batch), Cheetah (streaming), and
ElevenLabs Scribe (cloud).

We expose a uniform `TranscriptionResult` so the UI doesn't care which
engine produced it. Each word carries a confidence score in [0, 1] which
is the whole point of the debugger.

Engine priority (handled by the caller, not here):
    1. Picovoice (Leopard / Cheetah) — on-device, real word-level confidence
    2. ElevenLabs Scribe             — cloud fallback, per-word logprob
    3. Mock                          — offline simulation
"""

from __future__ import annotations

import io
import math
import os
import wave
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from config import AUDIO


@dataclass
class Word:
    word: str
    confidence: float
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class TranscriptionResult:
    transcript: str
    words: List[Word] = field(default_factory=list)
    engine: str = "leopard"

    @property
    def average_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(sum(w.confidence for w in self.words) / len(self.words))

    @property
    def min_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(min(w.confidence for w in self.words))


class TranscriberError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Leopard (batch)
# ---------------------------------------------------------------------------


class LeopardTranscriber:
    """Wraps `pvleopard` - batch mode. Feed it a full int16 utterance."""

    def __init__(self, access_key: str):
        if not access_key:
            raise TranscriberError("Missing Picovoice AccessKey.")
        try:
            import pvleopard
        except Exception as exc:
            raise TranscriberError(
                "pvleopard is not installed. Run `pip install pvleopard`."
            ) from exc

        try:
            self._leopard = pvleopard.create(access_key=access_key)
        except Exception as exc:
            raise TranscriberError(f"Failed to initialise Leopard: {exc}") from exc

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        audio = audio.flatten()
        if audio.size == 0:
            return TranscriptionResult(transcript="", words=[], engine="leopard")

        try:
            transcript, words = self._leopard.process(audio)
        except Exception as exc:
            raise TranscriberError(f"Leopard processing failed: {exc}") from exc

        parsed: List[Word] = []
        for w in words or []:
            parsed.append(
                Word(
                    word=getattr(w, "word", ""),
                    confidence=float(getattr(w, "confidence", 0.0)),
                    start_sec=float(getattr(w, "start_sec", 0.0)),
                    end_sec=float(getattr(w, "end_sec", 0.0)),
                )
            )

        return TranscriptionResult(
            transcript=transcript or "",
            words=parsed,
            engine="leopard",
        )

    def delete(self) -> None:
        try:
            self._leopard.delete()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cheetah (streaming) - v2 upgrade
# ---------------------------------------------------------------------------


class CheetahTranscriber:
    """Wraps `pvcheetah` - streaming mode. Feed it fixed-size frames."""

    def __init__(self, access_key: str, endpoint_duration_sec: float = 1.0):
        if not access_key:
            raise TranscriberError("Missing Picovoice AccessKey.")
        try:
            import pvcheetah
        except Exception as exc:
            raise TranscriberError(
                "pvcheetah is not installed. Run `pip install pvcheetah`."
            ) from exc

        try:
            self._cheetah = pvcheetah.create(
                access_key=access_key,
                endpoint_duration_sec=endpoint_duration_sec,
                enable_automatic_punctuation=True,
            )
        except Exception as exc:
            raise TranscriberError(f"Failed to initialise Cheetah: {exc}") from exc

        self.frame_length: int = int(self._cheetah.frame_length)
        self.sample_rate: int = int(self._cheetah.sample_rate)

    def process_stream(self, audio: np.ndarray) -> TranscriptionResult:
        """Process a full utterance in streaming chunks and return the result."""
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        audio = audio.flatten()

        transcript_parts: List[str] = []
        words: List[Word] = []

        for start in range(0, len(audio) - self.frame_length + 1, self.frame_length):
            frame = audio[start : start + self.frame_length]
            try:
                partial, is_endpoint = self._cheetah.process(frame)
            except Exception as exc:
                raise TranscriberError(f"Cheetah processing failed: {exc}") from exc
            if partial:
                transcript_parts.append(partial)
            if is_endpoint:
                try:
                    tail = self._cheetah.flush()
                except Exception as exc:
                    raise TranscriberError(f"Cheetah flush failed: {exc}") from exc
                if tail:
                    transcript_parts.append(tail)

        try:
            tail = self._cheetah.flush()
        except Exception as exc:
            raise TranscriberError(f"Cheetah final flush failed: {exc}") from exc
        if tail:
            transcript_parts.append(tail)

        transcript = "".join(transcript_parts).strip()

        # Cheetah's public API does not expose per-word confidence the way
        # Leopard does. We still produce a word list (with confidence=1.0 as
        # a placeholder) so the UI can render something; the debugger value
        # is clearer in Leopard mode.
        for token in transcript.split():
            words.append(Word(word=token.strip(".,!?"), confidence=1.0))

        return TranscriptionResult(
            transcript=transcript,
            words=words,
            engine="cheetah",
        )

    def delete(self) -> None:
        try:
            self._cheetah.delete()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ElevenLabs Scribe (cloud) - fallback when no Picovoice AccessKey is present
# ---------------------------------------------------------------------------


class ElevenLabsTranscriber:
    """Wraps the ElevenLabs Scribe STT API.

    Unlike Picovoice (which runs on-device), this engine is cloud-based:
    the audio is uploaded to ElevenLabs and the transcript comes back with
    per-word timestamps and — when the model exposes them — per-word
    log-probabilities that we convert to a confidence score in [0, 1].

    We keep the same surface area (`transcribe(np.ndarray) -> TranscriptionResult`)
    so the debugger UI can treat it like any other batch engine.
    """

    _ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
    _DEFAULT_MODEL = "scribe_v1"
    _REQUEST_TIMEOUT_SEC = 60.0

    def __init__(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        language_code: Optional[str] = None,
    ):
        if not api_key:
            raise TranscriberError(
                "Missing ElevenLabs API key. Set ELEVENLABS_API_KEY in your .env."
            )
        try:
            import requests  # noqa: F401
        except Exception as exc:
            raise TranscriberError(
                "`requests` is required for the ElevenLabs engine. "
                "Run `pip install requests`."
            ) from exc
        self._api_key = api_key.strip()
        self._model_id = model_id or self._DEFAULT_MODEL
        self._language = language_code

    @staticmethod
    def _encode_wav(audio: np.ndarray) -> bytes:
        """Encode a mono int16 numpy array as an in-memory WAV file."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(AUDIO.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(AUDIO.sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        import requests

        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        audio = audio.flatten()
        if audio.size == 0:
            return TranscriptionResult(transcript="", words=[], engine="elevenlabs")

        wav_bytes = self._encode_wav(audio)

        headers = {"xi-api-key": self._api_key}
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data: dict = {
            "model_id": self._model_id,
            "timestamps_granularity": "word",
        }
        if self._language:
            data["language_code"] = self._language

        try:
            resp = requests.post(
                self._ENDPOINT,
                headers=headers,
                files=files,
                data=data,
                timeout=self._REQUEST_TIMEOUT_SEC,
            )
        except Exception as exc:
            raise TranscriberError(
                f"ElevenLabs request failed (network error): {exc}"
            ) from exc

        if resp.status_code == 401:
            raise TranscriberError(
                "ElevenLabs rejected the API key (401). "
                "Check ELEVENLABS_API_KEY in your .env."
            )
        if resp.status_code == 429:
            raise TranscriberError(
                "ElevenLabs rate-limited this request (429). "
                "Slow down or upgrade your plan."
            )
        if resp.status_code >= 400:
            try:
                err = resp.json()
            except Exception:
                err = (resp.text or "")[:300]
            raise TranscriberError(
                f"ElevenLabs API error {resp.status_code}: {err}"
            )

        try:
            payload = resp.json()
        except Exception as exc:
            raise TranscriberError(
                f"ElevenLabs returned non-JSON response: {exc}"
            ) from exc

        transcript = (payload.get("text") or "").strip()
        raw_words = payload.get("words") or []

        words: List[Word] = []
        for w in raw_words:
            # Scribe emits `word`, `spacing`, and `audio_event` types; we only
            # surface real words in the debugger timeline.
            w_type = w.get("type", "word")
            if w_type != "word":
                continue
            token = (w.get("text") or "").strip()
            if not token:
                continue
            start = float(w.get("start") or 0.0)
            end = float(w.get("end") or start)
            confidence = self._word_confidence(w)
            words.append(
                Word(
                    word=token,
                    confidence=confidence,
                    start_sec=start,
                    end_sec=end,
                )
            )

        return TranscriptionResult(
            transcript=transcript,
            words=words,
            engine="elevenlabs",
        )

    @staticmethod
    def _word_confidence(word_obj: dict) -> float:
        """Map ElevenLabs per-word metadata to a [0, 1] confidence.

        Scribe sometimes exposes `logprob` (natural-log probability).
        If absent, we fall back to 1.0 so the UI still has a number to show.
        """
        logprob = word_obj.get("logprob")
        if logprob is None:
            logprob = word_obj.get("log_prob")
        if logprob is None:
            return 1.0
        try:
            conf = math.exp(float(logprob))
        except Exception:
            return 1.0
        if not math.isfinite(conf):
            return 1.0
        return float(max(0.0, min(1.0, conf)))

    def delete(self) -> None:  # parity with other engines
        return None


# ---------------------------------------------------------------------------
# Mock (offline demo, no AccessKey)
# ---------------------------------------------------------------------------


class MockTranscriber:
    """Offline demo transcriber. No AccessKey, no network, no real ASR.

    Generates a plausible-looking transcript with per-word confidence
    scores derived from the actual audio waveform (energy envelope).
    Use this to exercise the debugger UI without signing up for
    Picovoice. Swap to Leopard for real speech recognition.
    """

    _SAMPLE_PHRASES = [
        "hello world this is a quick test of the voice debugger",
        "the quick brown fox jumps over the lazy dog",
        "picovoice runs entirely on device with no cloud dependency",
        "set an alarm for seven in the morning please",
        "turn on the kitchen lights and start the coffee maker",
        "streaming speech to text with word level confidence",
        "debugging voice models is much easier when you can see the scores",
    ]

    def __init__(self, access_key: str | None = None, seed: Optional[int] = None):
        # access_key is accepted for API symmetry but ignored.
        self._rng = np.random.default_rng(seed)

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        audio = audio.flatten()

        if audio.size == 0:
            return TranscriptionResult(transcript="", words=[], engine="mock")

        duration = audio.size / AUDIO.sample_rate

        # Pick a phrase sized roughly to the audio duration (~2.5 words/sec).
        target_words = max(3, min(16, int(round(duration * 2.5))))
        phrase = self._pick_phrase(target_words)
        tokens = phrase.split()

        # Build an energy envelope over fixed windows to modulate confidence.
        window = max(1, int(AUDIO.sample_rate * 0.05))  # 50 ms windows
        # Normalise to [0, 1].
        abs_audio = np.abs(audio.astype(np.float32))
        if abs_audio.size < window:
            envelope = np.array([abs_audio.mean()], dtype=np.float32)
        else:
            trimmed = abs_audio[: (abs_audio.size // window) * window]
            envelope = trimmed.reshape(-1, window).mean(axis=1)
        peak = float(envelope.max()) if envelope.size else 1.0
        if peak <= 1e-6:
            envelope = np.zeros_like(envelope)
        else:
            envelope = envelope / peak

        # Overall "signal health" - louder, cleaner audio -> higher base conf.
        rms = float(np.sqrt(np.mean((audio.astype(np.float32)) ** 2))) / 32768.0
        base = 0.50 + min(0.20, rms * 3.0)  # roughly 0.50 - 0.70

        words: List[Word] = []
        step = duration / max(1, len(tokens))
        n = len(tokens)
        for i, tok in enumerate(tokens):
            start = i * step
            end = start + step
            # Map this word to a slice of the envelope.
            if envelope.size > 0:
                lo = int((i / n) * envelope.size)
                hi = max(lo + 1, int(((i + 1) / n) * envelope.size))
                slice_mean = float(envelope[lo:hi].mean()) if hi > lo else 0.0
            else:
                slice_mean = 0.0

            # Edge penalty: first and last words dip - matches the PFD insight.
            if i == 0 or i == n - 1:
                edge = -0.18
            else:
                edge = 0.0

            noise = float(self._rng.normal(0.0, 0.07))
            conf = base + 0.20 * slice_mean + edge + noise
            conf = float(np.clip(conf, 0.05, 0.98))

            words.append(
                Word(
                    word=tok,
                    confidence=conf,
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                )
            )

        transcript = " ".join(w.word for w in words)
        return TranscriptionResult(transcript=transcript, words=words, engine="mock")

    def _pick_phrase(self, target_words: int) -> str:
        # Pick the phrase whose length is closest to target_words.
        phrases = self._SAMPLE_PHRASES
        ranked = sorted(phrases, key=lambda p: abs(len(p.split()) - target_words))
        # Slight randomness so repeated runs vary.
        top = ranked[: min(3, len(ranked))]
        return str(self._rng.choice(top))

    def delete(self) -> None:  # parity with other engines
        return None


def make_transcriber(
    engine: str,
    access_key: str | None,
    eleven_api_key: str | None = None,
):
    """Construct a transcriber for the given engine id.

    Priority, when letting this function pick automatically via `resolve_engine`:
        Picovoice (leopard/cheetah) > ElevenLabs (elevenlabs) > Mock.

    For ElevenLabs, `eleven_api_key` takes precedence; if not supplied, the
    ELEVENLABS_API_KEY environment variable is used.
    """
    engine = (engine or "leopard").lower()
    if engine == "mock":
        return MockTranscriber(access_key=access_key)
    if engine == "leopard":
        return LeopardTranscriber(access_key=access_key or "")
    if engine == "cheetah":
        return CheetahTranscriber(access_key=access_key or "")
    if engine in ("elevenlabs", "eleven", "scribe"):
        key = (eleven_api_key or os.environ.get("ELEVENLABS_API_KEY") or "").strip()
        return ElevenLabsTranscriber(api_key=key)
    raise TranscriberError(
        f"Unknown engine '{engine}'. "
        "Use 'leopard', 'cheetah', 'elevenlabs', or 'mock'."
    )


def resolve_engine(
    has_picovoice_key: bool,
    has_eleven_key: bool,
    preferred: str | None = None,
) -> str:
    """Pick an engine based on the priority Picovoice > ElevenLabs > Mock.

    If `preferred` is set and viable given the keys, it is honoured.
    """
    if preferred:
        p = preferred.lower()
        if p in ("leopard", "cheetah") and has_picovoice_key:
            return p
        if p in ("elevenlabs", "eleven", "scribe") and has_eleven_key:
            return "elevenlabs"
        if p == "mock":
            return "mock"
    if has_picovoice_key:
        return "leopard"
    if has_eleven_key:
        return "elevenlabs"
    return "mock"
