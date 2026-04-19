"""Voice AI Debugger - Streamlit UI.

Run with:
    streamlit run app.py

You need a Picovoice AccessKey in the environment variable
PICOVOICE_ACCESS_KEY (or in a local .env file next to this script).
Grab one for free at https://console.picovoice.ai.
"""

from __future__ import annotations

import html
import time
from dataclasses import asdict
from typing import List, Optional

import numpy as np
import streamlit as st

from config import AUDIO, CONFIDENCE, get_access_key, get_elevenlabs_api_key
from recorder import (
    RecorderError,
    list_input_devices,
    load_wav_file,
    record_fixed_duration,
)
from transcriber import (
    TranscriberError,
    TranscriptionResult,
    Word,
    make_transcriber,
    resolve_engine,
)
from vad import CobraVAD, VADError
from diagnostics import (
    analyze_audio,
    append_failure,
    clear_failures,
    inject_white_noise,
    load_failure_audio,
    load_failures,
    position_confidence,
    word_error_rate,
)
from benchmarks import (
    benchmark_engine,
    rtf_verdict,
)
from visuals import (
    audio_to_wav_bytes,
    build_waveform_chart,
    homophone_tag,
    transcript_stability,
    word_slice_wav_bytes,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Voice AI Debugger",
    page_icon="🎙️",
    layout="centered",
)

st.markdown(
    """
    <style>
      .va-transcript {
        font-size: 1.25rem;
        line-height: 1.9;
        padding: 1rem 1.1rem;
        border-radius: 10px;
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
      }
      .va-word {
        padding: 2px 6px;
        margin: 0 2px;
        border-radius: 6px;
        font-weight: 500;
      }
      .va-word-high   { color: #1b873a; }
      .va-word-mid    { color: #b6791b; background: rgba(245, 180, 40, 0.15); }
      .va-word-low    {
        color: #b3261e;
        background: rgba(200, 50, 50, 0.15);
        border: 1px dashed rgba(200, 50, 50, 0.55);
      }
      .va-muted { color: #888; font-style: italic; }
      .va-homophone {
        font-size: 0.70rem;
        color: #6a4cff;
        background: rgba(106, 76, 255, 0.10);
        border: 1px solid rgba(106, 76, 255, 0.35);
        padding: 1px 5px;
        border-radius: 4px;
        margin-left: 4px;
        vertical-align: middle;
        font-weight: 600;
      }
      .va-compare-col {
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        background: rgba(120, 120, 120, 0.06);
        border: 1px solid rgba(120, 120, 120, 0.16);
        min-height: 140px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_log" not in st.session_state:
    st.session_state.session_log = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "noise_lab_results" not in st.session_state:
    st.session_state.noise_lab_results = []
if "consistency_takes" not in st.session_state:
    st.session_state.consistency_takes = []  # list of dicts: audio, transcript, words, avg_conf
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None
if "bench_results" not in st.session_state:
    st.session_state.bench_results = {}
if "replay_rows" not in st.session_state:
    st.session_state.replay_rows = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confidence_class(conf: float) -> str:
    if conf >= CONFIDENCE.high:
        return "va-word-high"
    if conf >= CONFIDENCE.low:
        return "va-word-mid"
    return "va-word-low"


def render_colored_transcript(words: List[Word], show_homophones: bool = True) -> str:
    if not words:
        return "<span class='va-muted'>No words detected.</span>"
    pieces = []
    for w in words:
        klass = _confidence_class(w.confidence)
        flag = " ⚠️" if w.confidence < CONFIDENCE.low else ""
        homo = homophone_tag(w.word) if show_homophones else None
        homo_badge = (
            f"<span class='va-homophone' title='known-ambiguous token'>{html.escape(homo)}</span>"
            if homo
            else ""
        )
        pieces.append(
            f"<span class='va-word {klass}' title='confidence={w.confidence:.2f}'>"
            f"{html.escape(w.word)}{flag}</span>{homo_badge}"
        )
    return " ".join(pieces)


def format_confidence(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_session_report() -> str:
    """Build a portable Markdown report of the current session."""
    from datetime import datetime

    lines: List[str] = []
    lines.append("# Voice AI Debugger — session report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")

    # Active configuration
    lines.append("## Configuration")
    lines.append(f"- Engine: `{st.session_state.get('last_meta', {}).get('engine', '—')}`")
    lines.append(f"- Warning threshold: {st.session_state.get('threshold_value', 0.75):.2f}")
    has_key_now = bool(st.session_state.get('has_key_flag'))
    has_eleven_now = bool(st.session_state.get('has_eleven_flag'))
    if has_key_now:
        key_state = "Picovoice (on-device)"
    elif has_eleven_now:
        key_state = "ElevenLabs Scribe (cloud fallback)"
    else:
        key_state = "none — mock mode"
    lines.append(f"- Active engine family: {key_state}")
    lines.append("")

    # Last result
    last_result = st.session_state.get("last_result")
    last_meta = st.session_state.get("last_meta", {}) or {}
    if last_result is not None:
        lines.append("## Last transcription")
        lines.append(f"- Transcript: **{last_result.transcript or '(empty)'}**")
        lines.append(f"- Average confidence: {last_result.average_confidence * 100:.1f}%")
        lines.append(f"- Minimum confidence: {last_result.min_confidence * 100:.1f}%")
        lines.append(f"- Word count: {len(last_result.words)}")
        lines.append(f"- Elapsed: {last_meta.get('elapsed_sec', 0.0):.2f}s")
        if last_result.words:
            lines.append("")
            lines.append("| word | confidence | start (s) | end (s) | ambiguity |")
            lines.append("|---|---|---|---|---|")
            for w in last_result.words:
                lines.append(
                    f"| {w.word} | {w.confidence:.2f} | {w.start_sec:.2f} | "
                    f"{w.end_sec:.2f} | {homophone_tag(w.word) or ''} |"
                )
        lines.append("")

    # Audio quality
    last_audio_local = st.session_state.get("last_audio")
    if last_audio_local is not None:
        stats = analyze_audio(last_audio_local)
        lines.append("## Audio quality")
        for k, v in stats.as_dict().items():
            if k != "warnings":
                lines.append(f"- {k}: {v}")
        if stats.warnings:
            lines.append("")
            lines.append("**Warnings:**")
            for w in stats.warnings:
                lines.append(f"- {w}")
        lines.append("")

    # Noise lab
    noise_rows = st.session_state.get("noise_lab_results") or []
    if noise_rows:
        lines.append("## Noise robustness sweep")
        lines.append("")
        lines.append("| SNR (dB) | avg conf | min conf | words | transcript |")
        lines.append("|---|---|---|---|---|")
        for r in sorted(noise_rows, key=lambda x: -x["snr_db"]):
            lines.append(
                f"| {r['snr_db']:g} | {r['avg_confidence'] * 100:.1f}% | "
                f"{r['min_confidence'] * 100:.1f}% | {r['word_count']} | "
                f"{r['transcript'][:80]} |"
            )
        lines.append("")

    # Benchmark
    bench_results = st.session_state.get("bench_results") or {}
    if bench_results:
        lines.append("## Latency & footprint benchmark")
        lines.append("")
        lines.append("| engine | N | mean (ms) | P50 | P95 | P99 | RTF mean | RTF P95 | init RSS Δ (MiB) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for eng, res in bench_results.items():
            lat = res.latency
            lines.append(
                f"| {eng} | {res.iterations} | {lat.mean_ms:.1f} | "
                f"{lat.p50_ms:.1f} | {lat.p95_ms:.1f} | {lat.p99_ms:.1f} | "
                f"{res.rtf_mean:.2f}x | {res.rtf_p95:.2f}x | "
                f"{res.init_rss_delta_mb if res.init_rss_delta_mb is not None else '—'} |"
            )
        lines.append("")

    # Compare
    cmp = st.session_state.get("compare_results")
    if cmp and cmp.get("pair"):
        lines.append("## Engine A/B comparison")
        lines.append("")
        for eng, data in cmp["pair"].items():
            r = data["result"]
            lines.append(
                f"- **{eng}** — `{r.transcript or '(empty)'}` "
                f"(avg {r.average_confidence * 100:.1f}%, "
                f"{len(r.words)} words, {data['elapsed']:.2f}s)"
            )
        lines.append("")

    # Consistency
    takes = st.session_state.get("consistency_takes") or []
    if takes:
        lines.append("## Consistency lab")
        for i, take in enumerate(takes, start=1):
            lines.append(
                f"- Take {i}: `{take['transcript']}` "
                f"(avg {take['avg_confidence'] * 100:.1f}%)"
            )
        lines.append("")

    # Session log
    log = st.session_state.get("session_log") or []
    if log:
        lines.append("## Session log")
        lines.append("")
        lines.append("| # | engine | avg conf | words | elapsed | transcript |")
        lines.append("|---|---|---|---|---|---|")
        for i, row in enumerate(log, start=1):
            lines.append(
                f"| {i} | {row['engine']} | "
                f"{row['avg_confidence'] * 100:.1f}% | "
                f"{row['word_count']} | {row['elapsed_sec']:.2f}s | "
                f"{row['transcript'][:80]} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    access_key_env = get_access_key()
    eleven_key_env = get_elevenlabs_api_key()

    has_pico_env = bool(access_key_env)
    has_eleven_env = bool(eleven_key_env)

    if has_pico_env and has_eleven_env:
        st.success(
            "✅ Picovoice AccessKey **and** ElevenLabs API key loaded.\n\n"
            "Priority: **Picovoice → ElevenLabs → Mock**."
        )
    elif has_pico_env:
        st.success("✅ Picovoice AccessKey loaded from environment / .env")
    elif has_eleven_env:
        st.info(
            "🟣 No Picovoice AccessKey — falling back to **ElevenLabs Scribe** "
            "(cloud STT) for transcription.\n\n"
            "Add `PICOVOICE_ACCESS_KEY` to `.env` to switch to on-device "
            "Picovoice engines."
        )
    else:
        st.info(
            "🎭 No API keys detected — running in **mock mode**.\n\n"
            "Priority order is **Picovoice → ElevenLabs → Mock**. "
            "Drop a key into a `.env` file next to `app.py`:\n\n"
            "`PICOVOICE_ACCESS_KEY=your-picovoice-key`  \n"
            "`ELEVENLABS_API_KEY=your-elevenlabs-key`"
        )

    with st.expander("Override keys for this session", expanded=False):
        override_key = st.text_input(
            "Picovoice AccessKey",
            value="",
            type="password",
            help="Optional — overrides PICOVOICE_ACCESS_KEY for this session only.",
        )
        override_eleven = st.text_input(
            "ElevenLabs API key",
            value="",
            type="password",
            help="Optional — overrides ELEVENLABS_API_KEY for this session only.",
        )
    access_key = (override_key or access_key_env or "").strip()
    eleven_api_key = (override_eleven or eleven_key_env or "").strip()
    has_key = bool(access_key)
    has_eleven = bool(eleven_api_key)

    # Build engine options in priority order: Picovoice -> ElevenLabs -> Mock.
    engine_options: list[str] = []
    if has_key:
        engine_options.extend(["leopard", "cheetah"])
    if has_eleven:
        engine_options.append("elevenlabs")
    engine_options.append("mock")

    # Default = the first available in priority order.
    default_engine_index = 0

    engine = st.radio(
        "Engine",
        options=engine_options,
        index=default_engine_index,
        horizontal=True,
        help=(
            "leopard = Picovoice batch STT with real word-level confidence. "
            "cheetah = Picovoice streaming STT (on-device). "
            "elevenlabs = ElevenLabs Scribe (cloud, fallback when no Picovoice key). "
            "mock = offline demo (simulated transcript, audio-driven confidence)."
        ),
    )

    duration = st.slider(
        "Record duration (seconds)",
        min_value=2.0,
        max_value=15.0,
        value=float(AUDIO.default_duration_seconds),
        step=0.5,
    )

    threshold = st.slider(
        "Warning threshold (avg confidence)",
        min_value=0.50,
        max_value=0.95,
        value=float(CONFIDENCE.default_threshold),
        step=0.01,
    )
    st.session_state.threshold_value = threshold
    st.session_state.has_key_flag = has_key
    st.session_state.has_eleven_flag = has_eleven

    use_vad = st.toggle(
        "Use Cobra VAD (filter silence)",
        value=False,
        help="Drops silent frames before STT - architecturally correct pipeline.",
    )

    vad_threshold = st.slider(
        "VAD voice probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        disabled=not use_vad,
    )

    with st.expander("Audio devices", expanded=False):
        devices = list_input_devices()
        if not devices:
            st.caption("No input devices detected (or sounddevice unavailable).")
        else:
            for dev in devices:
                st.write(
                    f"**[{dev['index']}]** {dev['name']} "
                    f"— {dev['channels']}ch @ {int(dev['sample_rate'])} Hz"
                )

    st.divider()
    st.caption(
        "Export a markdown report of this session (metrics, transcripts, "
        "benchmarks, quality findings). Use it to justify a design decision "
        "or attach to a bug report."
    )
    # Build the markdown lazily; it's cheap and always reflects live state.
    try:
        report_md = build_session_report()
    except Exception:
        report_md = "# Voice AI Debugger session\n\n(Report generation failed.)"
    st.download_button(
        "📄 Download session report (.md)",
        data=report_md.encode("utf-8"),
        file_name="voice_debugger_session.md",
        mime="text/markdown",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎙️ Voice AI Debugger")
st.caption(
    "See exactly what your speech engine hears, how confident it is, "
    "and where it's failing."
)


# ---------------------------------------------------------------------------
# Input row
# ---------------------------------------------------------------------------

col_rec, col_upload = st.columns([1, 1])

def _engine_has_credentials(eng: str) -> bool:
    """Can the currently-selected engine actually run?"""
    eng = (eng or "").lower()
    if eng == "mock":
        return True
    if eng in ("leopard", "cheetah"):
        return bool(access_key)
    if eng == "elevenlabs":
        return bool(eleven_api_key)
    return False


with col_rec:
    can_record = _engine_has_credentials(engine)
    record_clicked = st.button(
        f"🔴 Record {duration:g}s",
        width="stretch",
        type="primary",
        disabled=not can_record,
    )

with col_upload:
    uploaded = st.file_uploader(
        "…or upload a 16 kHz mono WAV",
        type=["wav"],
        label_visibility="collapsed",
    )

if engine == "mock":
    st.caption(
        "Mock mode active — simulated transcript, real audio-driven confidence. "
        "Set `PICOVOICE_ACCESS_KEY` or `ELEVENLABS_API_KEY` in `.env` to use a "
        "real STT engine (priority: Picovoice → ElevenLabs → Mock)."
    )
elif engine == "elevenlabs":
    st.caption(
        "Using **ElevenLabs Scribe** (cloud STT) — audio is uploaded to "
        "ElevenLabs. Add `PICOVOICE_ACCESS_KEY` to `.env` to switch to "
        "on-device Picovoice."
    )
else:
    st.caption(f"Using Picovoice **{engine.capitalize()}** with your AccessKey.")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def transcribe_once(audio: np.ndarray) -> Optional[TranscriptionResult]:
    """Run the currently selected engine on a raw audio buffer and return the result.

    Used by the Noise Robustness Lab so we don't go through VAD or session-log
    side effects. Returns None on error (error is surfaced via st.error).
    """
    try:
        transcriber = make_transcriber(engine, access_key, eleven_api_key)
    except TranscriberError as exc:
        st.error(str(exc))
        return None
    try:
        if engine == "cheetah":
            return transcriber.process_stream(audio)
        return transcriber.transcribe(audio)
    except TranscriberError as exc:
        st.error(str(exc))
        return None
    finally:
        transcriber.delete()


def run_pipeline(audio: np.ndarray, source: str) -> None:
    raw_audio = audio.copy()
    meta: dict = {"source": source, "samples": int(audio.size)}
    t0 = time.perf_counter()

    if use_vad:
        if not access_key:
            st.warning(
                "Cobra VAD needs a Picovoice AccessKey — skipping VAD for this run."
            )
            vad = None
        else:
            try:
                vad = CobraVAD(access_key=access_key, threshold=vad_threshold)
            except VADError as exc:
                st.error(f"VAD init failed: {exc}")
                return
    else:
        vad = None

    if vad is not None:
        try:
            audio, voiced_ratio = vad.filter_voiced(audio)
            meta["voiced_ratio"] = voiced_ratio
        except VADError as exc:
            st.error(f"VAD processing failed: {exc}")
            vad.delete()
            return
        finally:
            vad.delete()

        if audio.size == 0:
            st.warning(
                "Cobra VAD filtered out every frame - no voice detected. "
                "Try lowering the VAD threshold or re-recording."
            )
            return

    try:
        transcriber = make_transcriber(engine, access_key, eleven_api_key)
    except TranscriberError as exc:
        st.error(str(exc))
        return

    try:
        if engine == "cheetah":
            result = transcriber.process_stream(audio)
        else:
            result = transcriber.transcribe(audio)
    except TranscriberError as exc:
        st.error(str(exc))
        transcriber.delete()
        return
    finally:
        transcriber.delete()

    meta["elapsed_sec"] = time.perf_counter() - t0
    meta["engine"] = engine
    st.session_state.last_result = result
    st.session_state.last_meta = meta
    st.session_state.last_audio = raw_audio
    st.session_state.noise_lab_results = []  # stale once audio changes
    st.session_state.compare_results = None  # ditto
    st.session_state.session_log.append(
        {
            "source": source,
            "engine": engine,
            "transcript": result.transcript,
            "avg_confidence": result.average_confidence,
            "min_confidence": result.min_confidence,
            "word_count": len(result.words),
            "elapsed_sec": meta["elapsed_sec"],
            "word_confidences": [float(w.confidence) for w in result.words],
        }
    )


_engine_ready = _engine_has_credentials(engine)


def _spinner_label_for(eng: str) -> str:
    if eng == "mock":
        return "Running mock transcriber…"
    if eng == "elevenlabs":
        return "Uploading to ElevenLabs Scribe…"
    return "Transcribing with Picovoice…"


if record_clicked and _engine_ready:
    with st.spinner(f"Recording {duration:g}s of audio…"):
        try:
            audio = record_fixed_duration(duration)
        except (RecorderError, ValueError) as exc:
            st.error(str(exc))
            audio = None
    if audio is not None:
        with st.spinner(_spinner_label_for(engine)):
            run_pipeline(audio, source="microphone")

if uploaded is not None and _engine_ready:
    tmp_path = f"/tmp/voice_debugger_upload_{int(time.time() * 1000)}.wav"
    with open(tmp_path, "wb") as fh:
        fh.write(uploaded.getbuffer())
    try:
        audio = load_wav_file(tmp_path)
    except RecorderError as exc:
        st.error(str(exc))
        audio = None
    if audio is not None:
        with st.spinner("Transcribing uploaded audio…"):
            run_pipeline(audio, source=f"upload:{uploaded.name}")


# ---------------------------------------------------------------------------
# Tabbed views
# ---------------------------------------------------------------------------

result: Optional[TranscriptionResult] = st.session_state.last_result
meta: dict = st.session_state.last_meta
last_audio: Optional[np.ndarray] = st.session_state.last_audio

st.divider()

(
    tab_debug,
    tab_quality,
    tab_latency,
    tab_noise,
    tab_compare,
    tab_consistency,
    tab_truth,
) = st.tabs(
    [
        "🎙 Debugger",
        "🔊 Audio Quality",
        "⏱ Latency & Footprint",
        "🌩 Noise Robustness Lab",
        "⚖ Engine A/B Compare",
        "🔁 Consistency Lab",
        "📝 Ground Truth + Regression",
    ]
)


# --- Tab 1: Debugger ------------------------------------------------------

with tab_debug:
    if result is None:
        st.info(
            "Record audio or upload a WAV to see the transcript and "
            "per-word confidence scores."
        )
    else:
        avg_conf = result.average_confidence
        min_conf = result.min_confidence

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Avg confidence", format_confidence(avg_conf))
        mcol2.metric("Min confidence", format_confidence(min_conf))
        mcol3.metric("Words", len(result.words))
        mcol4.metric("Elapsed", f"{meta.get('elapsed_sec', 0.0):.2f}s")

        if avg_conf < threshold:
            st.error(
                f"⚠️ Average confidence {format_confidence(avg_conf)} is below "
                f"your threshold of {format_confidence(threshold)}. The engine "
                "is unsure — check the **Audio Quality** tab to see if the "
                "mic signal itself is the problem."
            )
        else:
            st.success(
                f"✅ Average confidence {format_confidence(avg_conf)} is at or "
                f"above your threshold of {format_confidence(threshold)}."
            )

        st.subheader("Transcript")
        st.markdown(
            f"<div class='va-transcript'>{render_colored_transcript(result.words)}</div>",
            unsafe_allow_html=True,
        )
        if result.words and any(homophone_tag(w.word) for w in result.words):
            st.caption(
                "🟣 Purple badges mark known-ambiguous tokens "
                "(homophones, digit-words) — these are the words Picovoice "
                "most often confuses regardless of confidence score."
            )

        if not result.words and result.transcript:
            st.caption(
                "The current engine did not return per-word confidence — "
                "switch to Leopard for the full debugger view."
            )

        if "voiced_ratio" in meta:
            st.caption(
                f"Cobra VAD kept {meta['voiced_ratio'] * 100:.1f}% of frames as voiced."
            )

        # --- Waveform with confidence heatmap ----------------------------
        if last_audio is not None and result.words and any(
            w.end_sec > w.start_sec for w in result.words
        ):
            st.subheader("Waveform with confidence overlay")
            st.caption(
                "Green/amber/red bands show where in the audio the engine "
                "was confident vs unsure. Hover a band for word details."
            )
            try:
                chart = build_waveform_chart(last_audio, result.words)
                st.altair_chart(chart, width="stretch")
            except Exception as exc:
                st.caption(f"(Could not render chart: {exc})")

        # --- Full-clip audio player --------------------------------------
        if last_audio is not None:
            st.audio(
                audio_to_wav_bytes(last_audio),
                format="audio/wav",
            )

        # --- Per-word click-to-replay chips ------------------------------
        if last_audio is not None and result.words and any(
            w.end_sec > w.start_sec for w in result.words
        ):
            with st.expander("▶ Click a word to hear just that slice", expanded=False):
                chip_cols = st.columns(4)
                for i, w in enumerate(result.words):
                    wav_bytes = word_slice_wav_bytes(last_audio, w)
                    with chip_cols[i % 4]:
                        klass = _confidence_class(w.confidence)
                        label_color = {
                            "va-word-high": "#1b873a",
                            "va-word-mid": "#b6791b",
                            "va-word-low": "#b3261e",
                        }[klass]
                        st.markdown(
                            f"<div style='color:{label_color};font-weight:600'>"
                            f"{html.escape(w.word)} "
                            f"<span style='color:#888;font-weight:400'>"
                            f"· {w.confidence:.2f}</span></div>",
                            unsafe_allow_html=True,
                        )
                        if wav_bytes is not None:
                            st.audio(wav_bytes, format="audio/wav")
                        else:
                            st.caption("(no timestamp)")

        # --- Session pattern footer (edge-penalty, cross-session) -------
        log = st.session_state.session_log
        if len(log) >= 3:
            utterances = [row.get("word_confidences", []) for row in log]
            utterances = [u for u in utterances if u]
            if utterances:
                buckets = position_confidence(utterances, n_buckets=10)
                edge_vals = [v for k, v in buckets.items() if k in (0, 9)]
                mid_vals = [v for k, v in buckets.items() if 2 <= k <= 7]
                if edge_vals and mid_vals:
                    delta = float(np.mean(mid_vals) - np.mean(edge_vals))
                    if delta > 0.02:
                        st.caption(
                            f"🧭 Session pattern: middle-of-utterance words are "
                            f"{delta * 100:.1f} pp more confident than edge "
                            f"words across your {len(utterances)} utterances — "
                            "pad your utterances or trim with Cobra VAD."
                        )

        if result.words:
            with st.expander("Per-word confidence chart", expanded=False):
                chart_data = {
                    w.word or f"[{i}]": w.confidence
                    for i, w in enumerate(result.words)
                }
                st.bar_chart(chart_data, height=220)

            with st.expander("Raw word data", expanded=False):
                st.dataframe(
                    [
                        {
                            "word": w.word,
                            "confidence": round(w.confidence, 4),
                            "start_sec": round(w.start_sec, 3),
                            "end_sec": round(w.end_sec, 3),
                            "homophone": homophone_tag(w.word) or "",
                        }
                        for w in result.words
                    ],
                    width="stretch",
                )


# --- Tab 2: Audio Quality -------------------------------------------------

with tab_quality:
    st.markdown(
        "**Why this matters:** Picovoice silently accepts any 16 kHz mono "
        "PCM stream — clipping, DC offset, or a dead-quiet mic will quietly "
        "tank confidence and you'll blame the model. This panel inspects the "
        "audio itself."
    )

    if last_audio is None:
        st.info("Record or upload audio to analyse.")
    else:
        stats = analyze_audio(last_audio)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("RMS", f"{stats.rms_dbfs:.1f} dBFS")
        q2.metric("Peak", f"{stats.peak_dbfs:.1f} dBFS")
        q3.metric("Est. SNR", f"{stats.estimated_snr_db:.1f} dB")
        q4.metric("Clipping", f"{stats.clipping_ratio * 100:.2f}%")

        r1, r2, r3 = st.columns(3)
        r1.metric("Voiced frames", f"{stats.voiced_fraction * 100:.1f}%")
        r2.metric("DC offset", f"{stats.dc_offset:+.4f}")
        r3.metric("Duration", f"{stats.duration_sec:.2f}s")

        if stats.warnings:
            for w in stats.warnings:
                st.warning(w)
        else:
            st.success("Audio looks clean — no quality issues detected.")

        with st.expander("Waveform + energy envelope", expanded=False):
            float_audio = last_audio.astype(np.float32) / 32768.0
            # Downsample for display.
            target = 2000
            if float_audio.size > target:
                step = float_audio.size // target
                wave_preview = float_audio[::step][:target]
            else:
                wave_preview = float_audio
            st.line_chart(wave_preview, height=180)


# --- Tab 2b: Latency & Footprint -----------------------------------------

with tab_latency:
    st.markdown(
        "**Why this matters:** For real-time voice apps — games, phone "
        "calls, live captioning — **latency beats accuracy**. This tab "
        "answers the procurement question *\"can we use Picovoice in a "
        "live pipeline on this hardware?\"* by measuring P50/P95/P99 "
        "latency, real-time factor (RTF), and memory footprint across "
        "N iterations."
    )

    available_bench_engines: list[str] = []
    if access_key:
        available_bench_engines.extend(["leopard", "cheetah"])
    if eleven_api_key:
        available_bench_engines.append("elevenlabs")
    available_bench_engines.append("mock")

    if last_audio is None:
        st.info("Record or upload audio first, then run the benchmark.")
    else:
        bc1, bc2, bc3 = st.columns([1.2, 1, 1])
        with bc1:
            bench_engines = st.multiselect(
                "Engines to benchmark",
                options=available_bench_engines,
                default=[available_bench_engines[0]],
            )
        with bc2:
            iterations = st.number_input(
                "Iterations",
                min_value=3,
                max_value=100,
                value=10,
                step=1,
            )
        with bc3:
            warmup = st.number_input(
                "Warmup runs",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                help="Discarded before measurement to stabilise caches/JIT.",
            )

        if st.button("⏱ Run benchmark", type="primary", disabled=not bench_engines):
            audio_duration = last_audio.size / AUDIO.sample_rate
            results_map: dict = {}

            progress = st.progress(0.0, text="Benchmarking…")
            for idx, eng in enumerate(bench_engines):
                # Warmup
                for _ in range(int(warmup)):
                    try:
                        tmp = make_transcriber(eng, access_key, eleven_api_key)
                        if eng == "cheetah":
                            tmp.process_stream(last_audio)
                        else:
                            tmp.transcribe(last_audio)
                        tmp.delete()
                    except Exception:
                        break

                res = benchmark_engine(
                    make_transcriber_fn=lambda e=eng: make_transcriber(e, access_key, eleven_api_key),
                    audio=last_audio,
                    iterations=int(iterations),
                    audio_duration_sec=audio_duration,
                    streaming=(eng == "cheetah"),
                    engine_name=eng,
                )
                results_map[eng] = res
                progress.progress(
                    (idx + 1) / max(1, len(bench_engines)),
                    text=f"Done with {eng}",
                )
            progress.empty()
            st.session_state.bench_results = results_map

        bench_results = st.session_state.get("bench_results", {})
        if bench_results:
            st.subheader("Latency statistics (ms)")
            table_rows = []
            for eng, res in bench_results.items():
                lat = res.latency
                table_rows.append(
                    {
                        "engine": eng,
                        "N": res.iterations,
                        "mean": f"{lat.mean_ms:.1f}",
                        "P50": f"{lat.p50_ms:.1f}",
                        "P95": f"{lat.p95_ms:.1f}",
                        "P99": f"{lat.p99_ms:.1f}",
                        "min": f"{lat.min_ms:.1f}",
                        "max": f"{lat.max_ms:.1f}",
                        "stddev": f"{lat.stddev_ms:.1f}",
                    }
                )
            st.dataframe(table_rows, width="stretch", hide_index=True)

            st.subheader("Real-time factor (RTF)")
            st.caption(
                "RTF = processing time / audio duration. **RTF < 1.0 is "
                "required** for real-time use. A conservative shop targets "
                "RTF P95 < 0.5 to leave headroom for spikes."
            )
            rtf_rows = []
            for eng, res in bench_results.items():
                emoji_mean, verdict_mean = rtf_verdict(res.rtf_mean)
                emoji_p95, _ = rtf_verdict(res.rtf_p95)
                rtf_rows.append(
                    {
                        "engine": eng,
                        "RTF mean": f"{emoji_mean} {res.rtf_mean:.3f}x",
                        "RTF P95": f"{emoji_p95} {res.rtf_p95:.3f}x",
                        "verdict": verdict_mean,
                    }
                )
            st.dataframe(rtf_rows, width="stretch", hide_index=True)

            st.subheader("Footprint")
            foot_rows = []
            for eng, res in bench_results.items():
                foot_rows.append(
                    {
                        "engine": eng,
                        "init time (ms)": f"{res.init_ms:.1f}",
                        "init RSS delta (MiB)": (
                            f"{res.init_rss_delta_mb:.1f}"
                            if res.init_rss_delta_mb is not None
                            else "—"
                        ),
                        "peak RSS (MiB)": (
                            f"{res.peak_rss_mb:.1f}"
                            if res.peak_rss_mb is not None
                            else "—"
                        ),
                        "errors": len(res.errors),
                    }
                )
            st.dataframe(foot_rows, width="stretch", hide_index=True)
            st.caption(
                "Memory numbers are process-level RSS deltas, not model-"
                "only footprint — they're directional, not exact."
            )

            # Per-iteration latency chart per engine.
            with st.expander("Per-iteration latency samples", expanded=False):
                for eng, res in bench_results.items():
                    if not res.latency.samples_ms:
                        continue
                    st.markdown(f"**{eng}**")
                    st.bar_chart(
                        {
                            f"{i + 1}": v
                            for i, v in enumerate(res.latency.samples_ms)
                        },
                        height=160,
                    )


# --- Tab 3: Noise Robustness Lab -----------------------------------------

with tab_noise:
    st.markdown(
        "**Why this matters:** Picovoice works beautifully in quiet rooms and "
        "falls over in noisy ones — but there's no built-in way to "
        "quantify *where* that cliff is for your model. This lab injects "
        "calibrated white noise at multiple SNRs, re-runs the engine, and "
        "plots the confidence-decay curve."
    )

    if last_audio is None:
        st.info("Record or upload a clean utterance first, then come back here.")
    else:
        default_snrs = "30, 20, 15, 10, 5, 0"
        snr_input = st.text_input(
            "Target SNR values (dB, comma-separated)",
            value=default_snrs,
            help="Lower is noisier. 30 dB ≈ pristine, 0 dB = noise as loud as speech.",
        )

        if st.button("Run robustness sweep", type="primary"):
            try:
                snrs = [float(s.strip()) for s in snr_input.split(",") if s.strip()]
            except ValueError:
                st.error("Could not parse SNR values — use numbers like '20, 10, 5'.")
                snrs = []

            rng = np.random.default_rng(0)
            sweep_rows = []
            progress = st.progress(0.0, text="Running sweep…")
            for i, snr in enumerate(snrs):
                noisy = inject_white_noise(last_audio, target_snr_db=snr, rng=rng)
                r = transcribe_once(noisy)
                if r is None:
                    continue
                sweep_rows.append(
                    {
                        "snr_db": snr,
                        "avg_confidence": r.average_confidence,
                        "min_confidence": r.min_confidence,
                        "transcript": r.transcript,
                        "word_count": len(r.words),
                    }
                )
                progress.progress((i + 1) / max(1, len(snrs)), text=f"SNR {snr:g} dB")
            progress.empty()
            st.session_state.noise_lab_results = sweep_rows

        sweep_rows = st.session_state.noise_lab_results
        if sweep_rows:
            sweep_rows_sorted = sorted(sweep_rows, key=lambda r: -r["snr_db"])
            chart = {
                f"{row['snr_db']:g} dB": row["avg_confidence"]
                for row in sweep_rows_sorted
            }
            st.subheader("Confidence vs SNR")
            st.bar_chart(chart, height=260)

            st.subheader("Transcript at each SNR")
            st.dataframe(
                [
                    {
                        "SNR (dB)": row["snr_db"],
                        "avg_conf": f"{row['avg_confidence'] * 100:.1f}%",
                        "min_conf": f"{row['min_confidence'] * 100:.1f}%",
                        "words": row["word_count"],
                        "transcript": row["transcript"],
                    }
                    for row in sweep_rows_sorted
                ],
                width="stretch",
                hide_index=True,
            )

            # Pinpoint the breaking point: first SNR where avg_conf crosses threshold.
            breaks = [
                r for r in sweep_rows_sorted if r["avg_confidence"] < threshold
            ]
            if breaks:
                worst_survivor = next(
                    (
                        r for r in sweep_rows_sorted
                        if r["avg_confidence"] >= threshold
                    ),
                    None,
                )
                if worst_survivor is not None:
                    st.info(
                        f"🎯 **Breaking point:** confidence falls below your "
                        f"threshold of {format_confidence(threshold)} at "
                        f"**{breaks[0]['snr_db']:g} dB SNR** "
                        f"(last clean pass was at {worst_survivor['snr_db']:g} dB)."
                    )
                else:
                    st.warning(
                        "Every SNR in the sweep fell below your threshold — "
                        "the model is struggling even on the cleanest version."
                    )
            else:
                st.success("Model stayed above threshold across the entire sweep.")


# --- Tab 3b: Engine A/B Compare ------------------------------------------

with tab_compare:
    st.markdown(
        "**Why this matters:** Leopard (batch, more context) and Cheetah "
        "(streaming, less context) give different answers on the same audio. "
        "This tab runs both on your last recording and highlights the diff — "
        "the classic accuracy-vs-latency tradeoff made visible."
    )

    available_engines: list[str] = []
    if access_key:
        available_engines.extend(["leopard", "cheetah"])
    if eleven_api_key:
        available_engines.append("elevenlabs")
    available_engines.append("mock")

    if last_audio is None:
        st.info("Record or upload audio first.")
    elif len(available_engines) < 2:
        st.info(
            "Only one engine is available right now (mock). Set "
            "`PICOVOICE_ACCESS_KEY` or `ELEVENLABS_API_KEY` in `.env` to "
            "enable cross-engine comparison."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            engine_a = st.selectbox(
                "Engine A",
                options=available_engines,
                index=0,
                key="compare_engine_a",
            )
        with c2:
            engine_b = st.selectbox(
                "Engine B",
                options=available_engines,
                index=min(1, len(available_engines) - 1),
                key="compare_engine_b",
            )

        if st.button("⚖ Run comparison", type="primary", disabled=engine_a == engine_b):
            progress = st.progress(0.0, text=f"Running {engine_a}…")
            results_pair = {}
            for idx, eng in enumerate((engine_a, engine_b)):
                try:
                    t = make_transcriber(eng, access_key, eleven_api_key)
                except TranscriberError as exc:
                    st.error(f"{eng}: {exc}")
                    t = None
                if t is None:
                    continue
                t0 = time.perf_counter()
                try:
                    if eng == "cheetah":
                        r = t.process_stream(last_audio)
                    else:
                        r = t.transcribe(last_audio)
                    elapsed = time.perf_counter() - t0
                    results_pair[eng] = {
                        "result": r,
                        "elapsed": elapsed,
                    }
                except TranscriberError as exc:
                    st.error(f"{eng}: {exc}")
                finally:
                    t.delete()
                progress.progress((idx + 1) / 2.0, text=f"Done with {eng}")
            progress.empty()
            st.session_state.compare_results = {
                "engine_a": engine_a,
                "engine_b": engine_b,
                "pair": results_pair,
            }

        compare_results = st.session_state.compare_results
        if compare_results and compare_results["pair"]:
            eng_a = compare_results["engine_a"]
            eng_b = compare_results["engine_b"]
            pair = compare_results["pair"]

            col_a, col_b = st.columns(2)
            for col, eng in ((col_a, eng_a), (col_b, eng_b)):
                data = pair.get(eng)
                with col:
                    st.markdown(f"#### {eng}")
                    if not data:
                        st.caption("(failed)")
                        continue
                    r: TranscriptionResult = data["result"]
                    st.markdown(
                        f"<div class='va-compare-col'>"
                        f"{render_colored_transcript(r.words) if r.words else html.escape(r.transcript) or '<span class=va-muted>(empty)</span>'}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Avg conf", f"{r.average_confidence * 100:.1f}%")
                    mc2.metric("Words", len(r.words))
                    mc3.metric("Elapsed", f"{data['elapsed']:.2f}s")

            # Transcript diff: WER between A and B, treating A as reference.
            if eng_a in pair and eng_b in pair:
                r_a = pair[eng_a]["result"].transcript
                r_b = pair[eng_b]["result"].transcript
                if r_a and r_b:
                    wer = word_error_rate(r_a, r_b)
                    st.subheader("Alignment diff (A → B)")
                    diff_pieces = []
                    for op, a_word, b_word in wer.alignment:
                        if op == "match":
                            diff_pieces.append(
                                f"<span class='va-word va-word-high'>"
                                f"{html.escape(a_word or '')}</span>"
                            )
                        elif op == "sub":
                            diff_pieces.append(
                                f"<span class='va-word va-word-low'>"
                                f"{html.escape(a_word or '')} → "
                                f"{html.escape(b_word or '')}</span>"
                            )
                        elif op == "del":
                            diff_pieces.append(
                                f"<span class='va-word va-word-low' "
                                f"title='only in {eng_a}'>"
                                f"[{html.escape(a_word or '')}]</span>"
                            )
                        else:  # ins
                            diff_pieces.append(
                                f"<span class='va-word va-word-mid' "
                                f"title='only in {eng_b}'>"
                                f"+{html.escape(b_word or '')}</span>"
                            )
                    st.markdown(
                        f"<div class='va-transcript'>{' '.join(diff_pieces)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Disagreement rate (WER treating {eng_a} as reference): "
                        f"**{wer.wer * 100:.1f}%** · "
                        f"{wer.substitutions} subs · {wer.deletions} dels · "
                        f"{wer.insertions} ins"
                    )


# --- Tab 3c: Consistency Lab ---------------------------------------------

with tab_consistency:
    st.markdown(
        "**Why this matters:** A single transcript tells you nothing about "
        "*stability*. If you say the same phrase three times, does Picovoice "
        "give you three identical answers? This lab records N takes "
        "back-to-back and shows which tokens are stable vs which flip "
        "between takes — separating flukes from patterns."
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        n_takes = st.number_input(
            "Number of takes",
            min_value=2,
            max_value=8,
            value=3,
            step=1,
        )
    with cc2:
        take_duration = st.number_input(
            "Seconds per take",
            min_value=2.0,
            max_value=10.0,
            value=float(duration),
            step=0.5,
        )

    can_consistency = _engine_has_credentials(engine)
    cstart = st.button(
        f"🔁 Record {int(n_takes)} takes",
        type="primary",
        disabled=not can_consistency,
        key="consistency_start",
    )

    if cstart:
        st.session_state.consistency_takes = []
        slot = st.empty()
        for i in range(int(n_takes)):
            slot.info(f"Take {i + 1}/{int(n_takes)} — recording now, speak your phrase…")
            try:
                audio = record_fixed_duration(float(take_duration))
            except (RecorderError, ValueError) as exc:
                st.error(str(exc))
                break
            slot.info(f"Take {i + 1}/{int(n_takes)} — transcribing…")
            r = transcribe_once(audio)
            if r is None:
                break
            st.session_state.consistency_takes.append(
                {
                    "audio": audio,
                    "transcript": r.transcript,
                    "avg_confidence": r.average_confidence,
                    "word_count": len(r.words),
                }
            )
        slot.empty()

    takes = st.session_state.consistency_takes
    if takes:
        st.subheader(f"{len(takes)} takes recorded")
        for i, take in enumerate(takes, start=1):
            st.markdown(
                f"**Take {i}** — conf {take['avg_confidence'] * 100:.1f}% · "
                f"{take['word_count']} words"
            )
            st.markdown(
                f"<div class='va-transcript' style='font-size:1.05rem'>"
                f"{html.escape(take['transcript']) or '<span class=va-muted>(empty)</span>'}"
                f"</div>",
                unsafe_allow_html=True,
            )

        stability = transcript_stability([t["transcript"] for t in takes])
        if stability["tokens"]:
            st.subheader("Per-token stability")
            st.caption(
                f"**Stable ratio:** {stability['stable_ratio'] * 100:.1f}% of "
                f"tokens appeared in every take. Length stddev across takes: "
                f"{stability['token_variance']:.2f}."
            )
            rows = []
            n = stability["take_count"]
            for row in stability["tokens"]:
                rows.append(
                    {
                        "token": row["token"],
                        "takes": f"{row['count']}/{n}",
                        "stable": "✅" if row["count"] == n else "⚠️",
                        "seen_in": ", ".join(str(t + 1) for t in row["takes"]),
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)

        if st.button("Clear consistency takes"):
            st.session_state.consistency_takes = []
            st.rerun()


# --- Tab 4: Ground Truth + WER -------------------------------------------

with tab_truth:
    st.markdown(
        "**Why this matters:** Picovoice gives you confidence, not correctness. "
        "Confidence can be high *and wrong* (homophones, domain jargon). "
        "Enter what you actually said to get the real word error rate and "
        "save failed utterances to a local library for regression testing."
    )

    if result is None:
        st.info("Record something first, then type the ground truth here.")
    else:
        ref = st.text_input(
            "What did you actually say?",
            value="",
            placeholder="e.g. set an alarm for seven in the morning",
        )
        if ref:
            wer = word_error_rate(ref, result.transcript)
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("WER", f"{wer.wer * 100:.1f}%")
            w2.metric("Substitutions", wer.substitutions)
            w3.metric("Deletions", wer.deletions)
            w4.metric("Insertions", wer.insertions)

            st.subheader("Alignment")
            pieces = []
            for op, r_word, h_word in wer.alignment:
                if op == "match":
                    pieces.append(
                        f"<span class='va-word va-word-high'>"
                        f"{html.escape(r_word or '')}</span>"
                    )
                elif op == "sub":
                    pieces.append(
                        f"<span class='va-word va-word-low' "
                        f"title='heard: {html.escape(h_word or '')}'>"
                        f"{html.escape(r_word or '')} → "
                        f"{html.escape(h_word or '')}</span>"
                    )
                elif op == "del":
                    pieces.append(
                        f"<span class='va-word va-word-low' title='missing'>"
                        f"[{html.escape(r_word or '')}]</span>"
                    )
                elif op == "ins":
                    pieces.append(
                        f"<span class='va-word va-word-mid' title='inserted'>"
                        f"+{html.escape(h_word or '')}</span>"
                    )
            st.markdown(
                f"<div class='va-transcript'>{' '.join(pieces)}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Green = match · Red = substitution/deletion · Amber = insertion"
            )

            if wer.wer > 0.0 and st.button("💾 Save to failure library (with audio)"):
                stats = analyze_audio(last_audio) if last_audio is not None else None
                entry = {
                    "engine": result.engine,
                    "reference": ref,
                    "hypothesis": result.transcript,
                    "wer": wer.wer,
                    "avg_confidence": result.average_confidence,
                    "min_confidence": result.min_confidence,
                    "audio_stats": stats.as_dict() if stats else None,
                }
                path = append_failure(entry, audio_int16=last_audio)
                st.success(
                    f"Saved to `{path}`. The audio is stored alongside so "
                    "you can replay this against future Picovoice releases."
                )

    # --- Regression Replay ---------------------------------------------
    failures = load_failures()
    if failures:
        with st.expander(
            f"🔁 Regression Replay — failure library ({len(failures)} entries)",
            expanded=False,
        ):
            st.markdown(
                "Re-run every saved failure through the **currently selected "
                "engine** and compare against the original transcript. This "
                "is how you validate a new Picovoice release: fixed vs. "
                "regressed vs. unchanged, with WER delta."
            )

            replayable = [e for e in failures if e.get("audio_path")]
            st.caption(
                f"{len(replayable)} of {len(failures)} entries have stored "
                "audio and can be replayed."
            )

            replay_disabled = not replayable or not _engine_has_credentials(engine)
            if st.button(
                "▶ Run regression replay on current engine",
                type="primary",
                disabled=replay_disabled,
                key="regression_replay",
            ):
                progress = st.progress(0.0, text="Replaying failures…")
                replay_rows = []
                for i, entry in enumerate(replayable):
                    audio_i = load_failure_audio(entry)
                    if audio_i is None:
                        continue
                    r = transcribe_once(audio_i)
                    if r is None:
                        continue
                    new_wer = word_error_rate(entry.get("reference", ""), r.transcript)
                    old_wer = float(entry.get("wer", 0.0))
                    delta = new_wer.wer - old_wer
                    if new_wer.wer == 0.0 and old_wer > 0.0:
                        status = "✅ fixed"
                    elif delta < -0.01:
                        status = "✅ improved"
                    elif delta > 0.01:
                        status = "❌ regressed"
                    else:
                        status = "➖ unchanged"
                    replay_rows.append(
                        {
                            "id": entry.get("id", "")[:14],
                            "status": status,
                            "reference": entry.get("reference", "")[:60],
                            "old_hyp": entry.get("hypothesis", "")[:60],
                            "new_hyp": r.transcript[:60],
                            "old_WER": f"{old_wer * 100:.1f}%",
                            "new_WER": f"{new_wer.wer * 100:.1f}%",
                            "Δ": f"{delta * 100:+.1f}pp",
                        }
                    )
                    progress.progress((i + 1) / max(1, len(replayable)))
                progress.empty()
                st.session_state.replay_rows = replay_rows

            replay_rows = st.session_state.get("replay_rows")
            if replay_rows:
                st.subheader(f"Replay results — {engine}")
                summary = {
                    "fixed": sum(1 for r in replay_rows if r["status"].startswith("✅ fixed")),
                    "improved": sum(1 for r in replay_rows if r["status"].startswith("✅ improved")),
                    "regressed": sum(1 for r in replay_rows if r["status"].startswith("❌")),
                    "unchanged": sum(1 for r in replay_rows if r["status"].startswith("➖")),
                }
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("✅ Fixed", summary["fixed"])
                s2.metric("✅ Improved", summary["improved"])
                s3.metric("❌ Regressed", summary["regressed"])
                s4.metric("➖ Unchanged", summary["unchanged"])
                st.dataframe(replay_rows, width="stretch", hide_index=True)

            st.divider()
            st.subheader("Full failure library")
            st.dataframe(
                [
                    {
                        "id": e.get("id", "")[:14],
                        "saved_at": e.get("saved_at", "")[:19],
                        "engine": e.get("engine", ""),
                        "WER": f"{e.get('wer', 0.0) * 100:.1f}%",
                        "avg_conf": f"{e.get('avg_confidence', 0.0) * 100:.1f}%",
                        "audio?": "✅" if e.get("audio_path") else "—",
                        "reference": e.get("reference", ""),
                        "hypothesis": e.get("hypothesis", ""),
                    }
                    for e in failures
                ],
                width="stretch",
                hide_index=True,
            )

            if st.button("🗑 Clear failure library", key="clear_failures"):
                clear_failures()
                st.session_state.replay_rows = None
                st.rerun()


# (Session Insights edge-penalty folded into the Debugger tab footer below.)


# ---------------------------------------------------------------------------
# Session log (always at the bottom, under the tabs)
# ---------------------------------------------------------------------------

if st.session_state.session_log:
    st.divider()
    st.subheader("Session log")
    log_rows = []
    for i, row in enumerate(st.session_state.session_log, start=1):
        log_rows.append(
            {
                "#": i,
                "engine": row["engine"],
                "source": row["source"],
                "transcript": row["transcript"][:80]
                + ("…" if len(row["transcript"]) > 80 else ""),
                "avg_conf": f"{row['avg_confidence'] * 100:.1f}%",
                "min_conf": f"{row['min_confidence'] * 100:.1f}%",
                "words": row["word_count"],
                "elapsed": f"{row['elapsed_sec']:.2f}s",
            }
        )
    st.dataframe(log_rows, width="stretch", hide_index=True)

    if st.button("Clear session log"):
        st.session_state.session_log = []
        st.session_state.last_result = None
        st.session_state.last_meta = {}
        st.session_state.last_audio = None
        st.session_state.noise_lab_results = []
        st.session_state.compare_results = None
        st.session_state.consistency_takes = []
        st.session_state.bench_results = {}
        st.session_state.replay_rows = None
        st.rerun()
