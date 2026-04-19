"""Performance benchmarking utilities.

For real-time voice apps (Picovoice's flagship use case) latency and
real-time factor (RTF) matter more than accuracy. This module provides
engine-agnostic benchmarking helpers:

- `latency_stats(samples)` -> P50/P95/P99 + mean/stddev
- `measure_init_footprint(ctor)` -> init time + RSS delta + heap warmup
- `measure_run(callable)` -> elapsed time + optional time-to-first-token
- `rtf(elapsed, audio_duration)` -> real-time factor (1.0 = real-time)

Intentionally dependency-light: uses `time.perf_counter` for timing and
falls back to `resource` on POSIX when `psutil` isn't available. Memory
numbers are approximate — we report them with a caveat in the UI.
"""

from __future__ import annotations

import gc
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Memory probe
# ---------------------------------------------------------------------------


def _current_rss_mb() -> Optional[float]:
    """Return the current process RSS in MiB, or None if unavailable."""
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KiB - normalise to MiB.
        if platform.system() == "Darwin":
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    samples_ms: List[float]
    mean_ms: float
    stddev_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict:
        return {
            "n": len(self.samples_ms),
            "mean_ms": round(self.mean_ms, 2),
            "stddev_ms": round(self.stddev_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


def latency_stats(samples_ms: List[float]) -> LatencyStats:
    if not samples_ms:
        return LatencyStats([], 0, 0, 0, 0, 0, 0, 0)
    arr = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        samples_ms=list(samples_ms),
        mean_ms=float(arr.mean()),
        stddev_ms=float(arr.std(ddof=0)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


# ---------------------------------------------------------------------------
# Real-time factor
# ---------------------------------------------------------------------------


def rtf(elapsed_sec: float, audio_duration_sec: float) -> float:
    """Real-Time Factor: processing time / audio duration.

    RTF < 1.0  => faster than real-time (required for streaming).
    RTF = 1.0  => borderline.
    RTF > 1.0  => cannot keep up; not viable for live use.
    """
    if audio_duration_sec <= 0:
        return float("inf")
    return float(elapsed_sec / audio_duration_sec)


def rtf_verdict(value: float) -> Tuple[str, str]:
    """Return a (emoji, sentence) human-readable verdict for an RTF value."""
    if value == float("inf"):
        return ("⚠️", "Audio has zero duration - RTF undefined.")
    if value < 0.3:
        return ("✅", f"RTF {value:.2f}x — comfortable headroom for real-time use.")
    if value < 0.7:
        return ("✅", f"RTF {value:.2f}x — real-time viable.")
    if value < 1.0:
        return ("⚠️", f"RTF {value:.2f}x — borderline; spikes will cause buffering.")
    return ("❌", f"RTF {value:.2f}x — slower than real-time; not viable for live audio.")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    engine: str
    iterations: int
    audio_duration_sec: float
    latency: LatencyStats
    rtf_mean: float
    rtf_p95: float
    init_ms: float
    init_rss_delta_mb: Optional[float]
    peak_rss_mb: Optional[float]
    transcripts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "engine": self.engine,
            "iterations": self.iterations,
            "audio_duration_sec": round(self.audio_duration_sec, 3),
            "latency": self.latency.as_dict(),
            "rtf_mean": round(self.rtf_mean, 3),
            "rtf_p95": round(self.rtf_p95, 3),
            "init_ms": round(self.init_ms, 2),
            "init_rss_delta_mb": (
                round(self.init_rss_delta_mb, 2)
                if self.init_rss_delta_mb is not None
                else None
            ),
            "peak_rss_mb": (
                round(self.peak_rss_mb, 2) if self.peak_rss_mb is not None else None
            ),
            "errors": list(self.errors),
        }


def benchmark_engine(
    make_transcriber_fn: Callable[[], object],
    audio: np.ndarray,
    iterations: int,
    audio_duration_sec: float,
    streaming: bool = False,
    engine_name: str = "unknown",
) -> BenchmarkResult:
    """Benchmark a transcriber callable over `iterations` runs.

    `make_transcriber_fn` should return a fresh transcriber object each
    time it's called (so we measure init cost too). It must expose
    `.transcribe(audio)` or `.process_stream(audio)`, and `.delete()`.
    """
    # ---- init footprint ----
    gc.collect()
    rss_before = _current_rss_mb()
    t_init = time.perf_counter()
    try:
        tr = make_transcriber_fn()
    except Exception as exc:
        return BenchmarkResult(
            engine=engine_name,
            iterations=0,
            audio_duration_sec=audio_duration_sec,
            latency=latency_stats([]),
            rtf_mean=float("inf"),
            rtf_p95=float("inf"),
            init_ms=0.0,
            init_rss_delta_mb=None,
            peak_rss_mb=None,
            errors=[f"init failed: {exc}"],
        )
    init_ms = (time.perf_counter() - t_init) * 1000.0
    rss_after_init = _current_rss_mb()
    init_rss_delta = (
        rss_after_init - rss_before
        if (rss_before is not None and rss_after_init is not None)
        else None
    )

    # ---- iterations ----
    samples_ms: List[float] = []
    transcripts: List[str] = []
    errors: List[str] = []
    peak_rss = rss_after_init

    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                if streaming:
                    r = tr.process_stream(audio)  # type: ignore[attr-defined]
                else:
                    r = tr.transcribe(audio)      # type: ignore[attr-defined]
                elapsed = (time.perf_counter() - t0) * 1000.0
                samples_ms.append(elapsed)
                transcripts.append(getattr(r, "transcript", ""))
            except Exception as exc:
                errors.append(str(exc))
            # sample peak RSS
            cur = _current_rss_mb()
            if cur is not None and (peak_rss is None or cur > peak_rss):
                peak_rss = cur
    finally:
        try:
            tr.delete()  # type: ignore[attr-defined]
        except Exception:
            pass

    lat = latency_stats(samples_ms)
    rtf_samples = [s / 1000.0 / max(audio_duration_sec, 1e-6) for s in samples_ms]
    rtf_mean = float(np.mean(rtf_samples)) if rtf_samples else float("inf")
    rtf_p95 = float(np.percentile(rtf_samples, 95)) if rtf_samples else float("inf")

    return BenchmarkResult(
        engine=engine_name,
        iterations=len(samples_ms),
        audio_duration_sec=audio_duration_sec,
        latency=lat,
        rtf_mean=rtf_mean,
        rtf_p95=rtf_p95,
        init_ms=init_ms,
        init_rss_delta_mb=init_rss_delta,
        peak_rss_mb=peak_rss,
        transcripts=transcripts,
        errors=errors,
    )
