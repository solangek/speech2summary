# Speech2Summary

Turns spoken audio into a structured text summary:
- transcribes audio with Groq Whisper
- summarizes the transcript with Gemini
- saves or downloads the result as a `.txt` file

## What's in this repo?

- `streamlit_app.py` — web UI: record or upload audio, pick languages, get a structured summary, optionally sync to Google Drive.
- `main.py` — command-line recorder: records from your microphone, transcribes, summarizes, and saves a timestamped `.txt` file in `~/summaries/`.
- `requirements.txt` — Python dependencies for the Streamlit app.
- `packages.txt` — system packages for Streamlit Cloud (includes `ffmpeg`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API keys

The CLI reads from environment variables; the Streamlit app reads from `.streamlit/secrets.toml`.

| Key | Required by | Purpose |
|---|---|---|
| `GROQ_API_KEY` | CLI + Streamlit | Groq Whisper transcription |
| `GEMINI_API_KEY` | CLI + Streamlit | Gemini summarization. Accepts a single string or an **array of keys** for automatic quota rotation |
| `SUMMARIZATION_PROMPT` | Streamlit (optional) | Assistant modes as a `[table]` of *mode → prompt*; overrides/extends the built-in `general` / `veterinarian` / `developer` modes. A legacy single string still works and appears as a `custom` mode |
| `GOOGLE_CLIENT_ID` | Streamlit (optional) | OAuth client ID for Google Drive sync |
| `GOOGLE_CLIENT_SECRET` | Streamlit (optional) | OAuth client secret |
| `GOOGLE_REDIRECT_URI` | Streamlit (optional) | OAuth redirect URI registered in Google Cloud console |

**CLI — set environment variables:**

```bash
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"
```

**Streamlit — `.streamlit/secrets.toml`:**

```toml
GROQ_API_KEY = "your_groq_key"
GEMINI_API_KEY = "your_gemini_key"

# Optional — only needed for Google Drive sync
GOOGLE_CLIENT_ID = "..."
GOOGLE_CLIENT_SECRET = "..."
GOOGLE_REDIRECT_URI = "https://your-app.streamlit.app/"

# Optional — assistant modes for the "Assistant type" selector.
# Overrides/extends the built-in mode.
# A TOML table must be the LAST thing in the file (otherwise the keys above
# get absorbed into it).
[SUMMARIZATION_PROMPT]
veterinarian = "Tu es docteur vétérinaire. Utilise la terminologie médicale française correcte..."
developer = "You are a software development assistant. Summarize this technical discussion..."
```

**Multiple Gemini keys for quota rotation (Streamlit):**

```toml
GEMINI_API_KEY = ["key1", "key2", "key3"]
```

On a quota error (429 / `RESOURCE_EXHAUSTED`), the app globally switches to the next key for all subsequent requests and cycles back to the first after the last key is used.

If the `GOOGLE_*` secrets are absent, the Drive sidebar section is hidden and the app works normally without Drive.

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the local URL in your browser, record or upload audio, and the summary will appear automatically.

### Sidebar options

| Option | Default | Effect |
|---|---|---|
| **Interface language** | Français | Switch the entire app UI between French and English |
| **Recording language** | Auto-detect | Language hint sent to Whisper. |
| **Summary language** | Français | Language Gemini uses to write the summary. |
| **Assistant type** | General assistant | Which summarization prompt to apply — general, or any custom mode from `secrets.toml` |
| **Summarize automatically** | ✅ on | Start transcription + summary as soon as audio is captured |
| **Include transcript in file** | ☐ off | Add the full transcript to the downloaded / Drive-saved file |
| **Compress audio** | ✅ on | Transcode audio ≥ 5 MB to mono 16 kHz Opus via `ffmpeg` before sending to Groq |
| **Connect Google Drive** | — | OAuth button, only visible when `GOOGLE_*` secrets are configured |

Models (Groq `whisper-large-v3-turbo`, Gemini `gemini-2.5-flash`) are fixed and not exposed in the UI.

### Supported audio / video upload formats

`wav`, `mp3`, `m4a`, `ogg`, `flac`, `webm`, **`m4v`**

`.m4v` and other video files have their audio track extracted with `ffmpeg` before transcription.

### App features

- **Result caching.** Once transcribed and summarized, the result is cached in the browser session. Clicking Download or any other Streamlit widget never retriggers transcription.
- **Audio compression.** Recordings ≥ 5 MB are transcoded with `ffmpeg` to mono 16 kHz Opus (~16 kbps) before upload. A 20-minute recording shrinks from ~110 MB to ~3 MB. Requires `ffmpeg`:
  - macOS: `brew install ffmpeg`
  - Streamlit Cloud: provided automatically via `packages.txt`
- **Video upload.** `.m4v` files are accepted; the audio track is extracted via `ffmpeg` before transcription.
- **Language control.** Transcription language hint and summary output language are independent settings. Auto-detect leaves language detection to Whisper.
- **UI localization.** The entire interface switches between French and English from a sidebar selector.
- **WebSocket keepalive.** Transcription and summarization run in background threads while a progress bar ticks every second, keeping the WebSocket alive through reverse-proxy idle timeouts.
- **Rate-limit retry.** Groq calls retry with exponential backoff on `RateLimitError`, honoring the `Retry-After` header.
- **Gemini key rotation.** Quota errors rotate to the next key globally for all subsequent requests.
- **Google Drive sync.** Optional OAuth flow saves summaries to a `Speech2Summary` folder on Drive and exposes a download history panel.
- **Assistant modes.** A sidebar **Assistant type** selector switches the summarization prompt between built-in modes (general, veterinarian, developer). Add or override modes by setting `SUMMARIZATION_PROMPT` as a `[table]` of *mode → prompt* in `secrets.toml`; a legacy single string still works as a `custom` mode.

### Usage tracing

The app emits one structured JSON log line per event to stdout. On Streamlit Cloud, view them under **Manage app → Logs**.

```json
{"ts":"2026-05-18T14:32:07+00:00","event":"transcribe_done","session_id":"a1b2c3","model":"whisper-large-v3-turbo","language":"fr","bytes":2987421,"duration_s":18.4,"transcript_chars":14203}
{"ts":"2026-05-18T14:32:15+00:00","event":"summarize_done","session_id":"a1b2c3","model":"gemini-2.5-flash","input_language":"French","output_language":"French","transcript_chars":14203,"summary_chars":812,"duration_s":7.1}
```

Events: `run_started`, `extract_audio_done` / `extract_audio_failed`, `compress_done` / `compress_skipped` / `compress_failed`, `transcribe_done` / `transcribe_failed` / `transcribe_empty`, `summarize_done` / `summarize_failed`, `drive_saved` / `drive_save_failed`.

## Run the CLI

```bash
python3 main.py
```

| Flag | Default | Description |
|---|---|---|
| `--output` | `~/summaries/` | Directory where `.txt` summaries are saved |
| `--groq-model` | `whisper-large-v3-turbo` | Groq transcription model |
| `--gemini-model` | `gemini-2.5-flash` | Gemini summarization model |

The CLI requires `numpy`, `sounddevice`, and `soundfile` (not in `requirements.txt`):

```bash
pip install numpy sounddevice soundfile
```
