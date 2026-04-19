# Voice AI Debugger

> A local, on-device debugger for [Picovoice](https://picovoice.ai)'s speech
> engines. Record an utterance, see the transcript, and watch per-word
> confidence scores live — so you can tune your models instead of guessing.

![engine](https://img.shields.io/badge/engine-Picovoice%20Leopard%20%7C%20Cheetah-8A2BE2)
![ui](https://img.shields.io/badge/ui-Streamlit-ff4b4b)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

## Why this exists

Picovoice ships powerful on-device STT (Leopard/Cheetah) that returns
word-level confidence scores — but there is no lightweight tool to watch
the engine's internal state while it runs. When a model works in testing
but breaks in production (different accents, background noise, mic
quality), developers are stuck guessing.

This debugger plugs directly into Picovoice so you can **see what the
engine hears and how sure it is about each word**. No cloud, no latency,
no data leaving your machine — the same values that make Picovoice
itself worth using.

## What it does

A seven-tab Streamlit app built around Picovoice's actual engines and
the diagnostic features that surface what Picovoice itself does not.
Features are triaged by real-world buyer/retention value, not by what
was easy to build.

### 🎙 Debugger (core)

- Captures microphone audio in real time at 16 kHz / mono / int16.
- Runs it through **Picovoice Leopard** (batch STT) or **Cheetah**
  (streaming STT). Falls back to a built-in `mock` engine when no
  AccessKey is present.
- Displays the live transcript with **color-coded words** by
  confidence:
  - ≥ 0.85 → green
  - 0.60 – 0.84 → amber
  - &lt; 0.60 → red + ⚠️ flag
- **🌊 Waveform + confidence heatmap** — colored bands overlaid on the
  actual audio waveform show exactly where the engine was unsure.
- **▶ Per-word click-to-replay chips** — every word becomes a tiny
  audio player so you can hear the exact slice Picovoice scored.
- **🟣 Homophone / ambiguity flagger** — inline purple badges on
  known-fragile tokens (`to/too/two`, `there/their`, digits, etc.)
  that Picovoice regularly confuses regardless of confidence.
- Avg / min confidence metrics, tunable **warning threshold** slider,
  per-word bar chart, raw word table (with homophone column).
- Optional **Cobra VAD** filter drops silent frames before STT.

### 🔊 Audio Quality Pre-Flight

**Flaw it targets:** Picovoice silently accepts any 16 kHz mono PCM
stream — clipping, DC offset, or a dead-quiet mic quietly tank
confidence and you'll blame the model.

Measures RMS / peak dBFS, clipping ratio, DC offset, estimated SNR,
and voiced fraction of the last audio, then emits actionable
warnings ("reduce mic gain", "DC offset detected", "SNR only 4 dB").

### ⏱ Latency & Footprint (procurement-grade)

**Flaw it targets:** For real-time voice apps — games, phone calls,
live captioning — **latency beats accuracy**, and Picovoice doesn't
publish P50/P95/P99 numbers or real-time-factor on your hardware.

Runs N iterations of any selected engine(s) on your last recording
with configurable warmup runs. Reports:

- Latency table (mean / P50 / P95 / P99 / min / max / stddev) in ms.
- **Real-Time Factor** (RTF) table with verdict emoji — `RTF < 1.0`
  is required for live use; a conservative shop targets P95 < 0.5.
- Init time, init RSS delta, peak RSS (via `psutil` when available;
  falls back to `resource.getrusage`).
- Per-iteration latency chart (spot cold-cache and GC spikes).

This is the tab that answers the hardware OEM's first question:
"will Picovoice run real-time on our device?"

### ⚖ Engine A/B Compare

**Flaw it targets:** Leopard (batch, more context) and Cheetah
(streaming, less context) give different answers on the same audio.
The accuracy-vs-latency tradeoff is real but invisible.

Pick any two engines (leopard / cheetah / mock), runs both on your
last recording, renders side-by-side transcripts with an aligned
**word-level diff** (same green/red/amber colour scheme as the
debugger). Shows the disagreement rate as a WER number.

### 🔁 Consistency Lab

**Flaw it targets:** A single transcript tells you nothing about
*stability*. Is this failure a fluke or a pattern?

Records N takes (2–8) of the same phrase back-to-back, runs each
through the engine, and computes **per-token stability**: which
words appeared in every take (stable ✅) vs which flipped between
takes (⚠️), plus transcript-length variance.

### 🌩 Noise Robustness Lab

**Flaw it targets:** No built-in way to quantify *where* your
model's noise cliff is.

Injects calibrated white noise at a configurable list of SNRs
(e.g. `30, 20, 15, 10, 5, 0` dB), re-runs the engine at each, plots
the confidence-decay curve, and pinpoints the **breaking point**
SNR where avg-confidence drops below your threshold. Turns "it
works on my laptop" into a number.

### 📝 Ground Truth + 🔁 Regression Replay

**Flaw it targets:** Confidence ≠ correctness. Picovoice can be
95% confident and transcribe the wrong homophone. There is no
feedback loop — and worse, no way to trust a new Picovoice release
without re-testing everything manually.

Type what you actually said; the app computes **WER** (word error
rate) with aligned substitutions / deletions / insertions and saves
failed utterances **with their audio** to `failure_library_audio/`
plus `failure_library.json`.

Then hit **▶ Run regression replay on current engine** to replay
every saved failure through the currently-selected engine and see a
summary: **fixed / improved / regressed / unchanged** with WER delta
per entry. This is how you validate a Picovoice SDK upgrade without
re-recording anything, and how you build organisational trust to keep
upgrading. This is the single most important feature for long-term
retention.

### 📄 Downloadable session report

**Flaw it targets:** You finish a debugging session, you have
findings, and… nothing leaves the browser. Decision-makers don't
see it.

Sidebar **Download session report (.md)** button exports the
entire session — configuration, last transcription with per-word
confidences, audio quality findings, noise sweep, benchmark results,
A/B comparison, consistency takes, session log — as a single
Markdown file. Paste into Notion / a PR / Slack and you've turned
"I played with a debugger" into "I can justify this decision."

### 🧭 Session pattern footer (compact)

The "edge-penalty" finding from the PFD (first/last words score
lower) now lives as a small caption under the transcript once
you have ≥3 utterances. It's a nice-to-know pattern, not a tab-
sized investment — demoted to its real value.

## Architecture

```
Mic input  →  Cobra VAD (optional)  →  Leopard / Cheetah STT
           →  Parse transcript + per-word confidence
           →  Streamlit UI (color-coded transcript, metrics, chart)
```

## File structure

```
voice-debugger/
├── app.py              # Streamlit UI — 5 tabs, main entry point
├── recorder.py         # sounddevice mic capture + WAV loader
├── transcriber.py      # Leopard / Cheetah / Mock wrappers
├── vad.py              # Cobra VAD integration (optional)
├── diagnostics.py      # Audio stats, noise injection, WER, failure library
├── benchmarks.py       # Latency percentiles, RTF, footprint probe
├── visuals.py          # Waveform chart, WAV slicing, homophones, stability
├── config.py           # AccessKey loader, thresholds, sample rate
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Credentials (all optional — priority-based fallback)

The debugger resolves its STT engine in this priority order:

1. **Picovoice** (`PICOVOICE_ACCESS_KEY`) — on-device, real word-level
   confidence. This is the preferred engine because the whole debugger
   is designed around Picovoice's per-word scores.
2. **ElevenLabs Scribe** (`ELEVENLABS_API_KEY`) — cloud fallback with
   per-word timestamps and (when the model exposes them) `logprob`-based
   confidence. Use this if you can't obtain a Picovoice AccessKey.
3. **Mock** — offline simulation. No network, no keys.

Any key you do provide enables the corresponding engine automatically:

- Only Picovoice set → uses Leopard (default).
- Only ElevenLabs set → uses Scribe.
- Both set → Picovoice is preferred, but you can switch to ElevenLabs
  from the sidebar (useful for A/B comparison).
- Neither set → mock mode.

Get keys:

- Picovoice: [console.picovoice.ai](https://console.picovoice.ai) —
  **a personal email works fine; no company email required.**
- ElevenLabs: [elevenlabs.io → API keys](https://elevenlabs.io/app/settings/api-keys).

### 2. Install

```bash
cd voice-debugger
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS / Linux you may also need PortAudio for `sounddevice`:

```bash
# macOS
brew install portaudio

# Debian/Ubuntu
sudo apt-get install -y libportaudio2
```

### 3. Configure your keys

```bash
cp .env.example .env
# then edit .env and paste whichever keys you have
```

`.env` (any subset is valid):

```bash
PICOVOICE_ACCESS_KEY=your-picovoice-key
ELEVENLABS_API_KEY=your-elevenlabs-key
```

Or export them for the current shell:

```bash
export PICOVOICE_ACCESS_KEY="your-picovoice-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

`.env` is git-ignored, so neither key will be committed.

### 4. Run

```bash
streamlit run app.py
```

Streamlit opens a browser tab at `http://localhost:8501`. Click
**🔴 Record**, speak, and watch the confidence scores land.

## Fallback modes (no Picovoice AccessKey)

### ElevenLabs Scribe (cloud)

If you have an ElevenLabs API key, the app automatically exposes the
`elevenlabs` engine. It:

- Encodes your 16 kHz mono audio as an in-memory WAV.
- Uploads it to `https://api.elevenlabs.io/v1/speech-to-text`
  (`scribe_v1` model, word-level timestamps).
- Maps any per-word `logprob` Scribe returns to a [0, 1] confidence,
  so every existing widget (coloring, threshold warnings, waveform
  heatmap, WER diff) works identically to Leopard.
- Surfaces real API errors (401, 429, transport) through the same
  `TranscriberError` channel.

Switch back to Picovoice by dropping a `PICOVOICE_ACCESS_KEY` into
`.env` — it is always preferred over ElevenLabs.

### Mock (fully offline)

If you have neither key, pick **`mock`** in the Engine selector. The
app will:

- Record real audio from your microphone.
- Fabricate a plausible transcript of roughly the right length.
- Compute per-word confidence scores from your **actual audio energy**,
  with a small penalty on the first and last words so the debugger UI
  surfaces the same "edges are less confident" pattern Leopard shows.

This is a UI demo — it is not real speech recognition — but every
widget in the app (transcript coloring, threshold warnings, chart, log)
behaves exactly as it will when you plug in a real engine.

## Quick smoke test (no UI)

Before opening Streamlit, confirm your key and mic work end-to-end:

```python
from config import get_access_key
from recorder import record_fixed_duration
from transcriber import LeopardTranscriber

audio = record_fixed_duration(5.0)
t = LeopardTranscriber(access_key=get_access_key())
result = t.transcribe(audio)
print(result.transcript)
for w in result.words:
    print(f"  {w.word:20s}  {w.confidence:.2f}")
```

## Notes on engines

| Engine      | Mode        | Location  | Word confidence               | Credential             | Best for                              |
| ----------- | ----------- | --------- | ----------------------------- | ---------------------- | ------------------------------------- |
| Leopard     | Batch STT   | On-device | ✅ Real                        | `PICOVOICE_ACCESS_KEY` | v1 debugger (default, **preferred**)  |
| Cheetah     | Streaming   | On-device | ⚠️ Limited (not exposed)      | `PICOVOICE_ACCESS_KEY` | Live-demo feel, latency comparisons   |
| ElevenLabs  | Batch STT   | Cloud     | ✅ `logprob`-derived           | `ELEVENLABS_API_KEY`   | Fallback when Picovoice key unavailable |
| mock        | Simulated   | Offline   | Synthetic (real audio-driven) | — none —               | UI exploration, offline demos         |
| Cobra (VAD) | VAD only    | On-device | n/a                           | `PICOVOICE_ACCESS_KEY` | Silence filtering                     |

Leopard is the recommended engine for the debugger because it exposes
the per-word confidence scores the tool is built to visualise. Cheetah
is supported as a v2 upgrade but its public API does not expose the
same per-word confidence — the debugger will still render the
transcript, just without the color-coding detail.

## Interview framing

> "Picovoice gives you powerful on-device STT, but there's no built-in
> way for developers to *see* confidence during development. I built a
> debugger that makes that visible — Leopard for transcription and
> scoring, Cobra for VAD, Streamlit for the UI. Everything runs locally,
> which aligns with Picovoice's core value proposition."

The interesting pattern the tool surfaces: confidence varies
dramatically by word position and phoneme — first and last words in an
utterance consistently score lower. That's the kind of thing you'd
never notice without this debugger.

## License

MIT — build on it, ship it, show it off.
