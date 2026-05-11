# Speech2Summary

Small project for turning spoken audio into a text summary:
- transcribe audio with Groq Whisper
- summarize the transcript with Gemini
- save or download the result as a text file

## What’s in this repo?

- `main.py` — command-line recorder: records from your microphone, transcribes, summarizes, and saves a timestamped `.txt` file in `~/summaries/` by default.
- `streamlit_app.py` — simple web UI for recording or uploading audio, then downloading the summary + transcript.
- `requirements.txt` — currently includes the Streamlit app dependencies.

## Setup

Create a virtual environment and install dependencies:

```bash
cd "/Users/solangekarsenty/workspace/speech2summary"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Secrets and environment variables

The CLI reads from environment variables; the Streamlit app reads from `.streamlit/secrets.toml`. The keys themselves are the same.

| Key | Required by | Purpose |
|---|---|---|
| `GROQ_API_KEY` | CLI + Streamlit | Groq Whisper transcription |
| `GEMINI_API_KEY` | CLI + Streamlit | Gemini summarization (the SDK also accepts `GOOGLE_API_KEY`; if both are set, `GOOGLE_API_KEY` wins) |
| `SUMMARIZATION_PROMPT` | Streamlit (optional) | Override the built-in meeting/lecture prompt. Falls back to `DEFAULT_PROMPT` in `streamlit_app.py` |
| `GOOGLE_CLIENT_ID` | Streamlit (optional, Drive only) | OAuth client ID for the Google Drive sync feature |
| `GOOGLE_CLIENT_SECRET` | Streamlit (optional, Drive only) | OAuth client secret |
| `GOOGLE_REDIRECT_URI` | Streamlit (optional, Drive only) | OAuth redirect URI registered in your Google Cloud console (e.g. `https://your-app.streamlit.app/`) |

CLI example:

```bash
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"
```

Streamlit example (`.streamlit/secrets.toml`):

```toml
GROQ_API_KEY = "your_groq_key"
GEMINI_API_KEY = "your_gemini_key"

# Optional — override the summarization prompt
SUMMARIZATION_PROMPT = """You are a meeting assistant. ..."""

# Optional — only needed to enable the Google Drive sync feature
GOOGLE_CLIENT_ID = "..."
GOOGLE_CLIENT_SECRET = "..."
GOOGLE_REDIRECT_URI = "https://your-app.streamlit.app/"
```

If the three `GOOGLE_*` secrets are absent, the Drive sidebar section is hidden and the rest of the app works normally.

## Run the CLI

```bash
python3 main.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--output` | `~/summaries/` | Directory where timestamped `.txt` summaries are saved |
| `--groq-model` | `whisper-large-v3-turbo` | Groq transcription model. Other supported values: `whisper-large-v3`, `distil-whisper-large-v3-en` |
| `--gemini-model` | `gemini-2.5-flash` | Gemini summarization model. Other supported values: `gemini-2.0-flash`, `gemini-1.5-flash` |
| `--no-compress` | off | Skip the `ffmpeg` Opus compression step (sends raw WAV to Groq) |
| `--log-file` | — | Append JSON usage events to this file in addition to stderr |

Environment variables honoured by the CLI: `GROQ_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), and optionally `SUMMARIZATION_PROMPT` to override the default summarization prompt.

JSON usage events are emitted to stderr (and optionally to `--log-file`) at every key step — same schema as the Streamlit app: `run_started`, `compress_done`/`compress_skipped`/`compress_failed`, `transcribe_done`/`transcribe_failed`/`transcribe_empty`, `summarize_done`/`summarize_failed`, `run_done`.

Example:

```bash
python3 main.py --output ~/summaries --groq-model whisper-large-v3-turbo --gemini-model gemini-2.5-flash
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL in your browser, record audio or upload a file, and click **Transcribe & Summarize**.

### Streamlit sidebar options

| Option | Default | Effect |
|---|---|---|
| **Transcription model (Groq)** | `whisper-large-v3-turbo` | Whisper model used for transcription. Choices: `whisper-large-v3-turbo`, `whisper-large-v3`, `distil-whisper-large-v3-en` (English-only, faster) |
| **Summarization model (Gemini)** | `gemini-2.5-flash` | Gemini model used for summarization. Choices: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash` |
| **Résumer automatiquement** | ✅ on | When on, transcription + summary start as soon as audio is captured. When off, a "Transcrire & Résumer" button is shown |
| **Inclure la transcription dans le fichier** | ☐ off | When on, the downloaded/Drive-saved text file includes the full transcript in addition to the summary |
| **Compresser l'audio avant transcription** | ✅ on | When on, audio ≥ 5 MB is transcoded to mono 16 kHz Opus (~16 kbps) via `ffmpeg` before upload. Disable to send the raw recording |
| **Connecter Google Drive** | — | OAuth sign-in button. Only visible when the three `GOOGLE_*` secrets are configured |

### Streamlit app features

- **Server-side audio compression.** Recordings ≥ 5 MB are transcoded with `ffmpeg` to mono 16 kHz Opus (~16 kbps) before being sent to Groq. A 20-minute recording shrinks from ~110 MB to ~3 MB, well below Groq's per-file limit. Toggle via the "Compresser l'audio avant transcription" checkbox in the sidebar. Requires `ffmpeg` on the host:
  - macOS: `brew install ffmpeg`
  - Streamlit Cloud: provided automatically via `packages.txt`
- **Websocket keepalive.** Transcription runs in a background thread while a progress bar ticks every second, keeping the websocket active through reverse-proxy idle timeouts.
- **Rate-limit retry.** Groq calls retry with exponential backoff on `RateLimitError`, honoring the `Retry-After` header.
- **Google Drive sync.** Optional OAuth flow to save summaries to a `Speech2Summary` folder on Drive and browse history. Needs these secrets:
  ```toml
  GOOGLE_CLIENT_ID = "..."
  GOOGLE_CLIENT_SECRET = "..."
  GOOGLE_REDIRECT_URI = "https://your-app/"
  ```
- **Custom summarization prompt.** Override the default by setting `SUMMARIZATION_PROMPT` in `secrets.toml`.

### Usage tracing

The app emits one structured JSON log line per meaningful event to stdout. On Streamlit Cloud, view them under **Manage app → Logs**. Example:

```json
{"ts":"2026-05-11T14:32:07+00:00","event":"transcribe_done","session_id":"a1b2c3d4e5f6","model":"whisper-large-v3-turbo","bytes":2987421,"duration_s":18.4,"transcript_chars":14203}
```

Events: `run_started`, `compress_done` / `compress_skipped` / `compress_failed`, `transcribe_done` / `transcribe_failed` / `transcribe_empty`, `summarize_done` / `summarize_failed`, `drive_saved` / `drive_save_failed`. A random `session_id` ties events from the same browser session together. No PII is logged.

## Note about CLI dependencies

`main.py` imports `numpy`, `sounddevice`, and `soundfile`, but they are not currently listed in `requirements.txt`.
If you want to use the CLI recorder, you may need to install them manually:

```bash
pip install numpy sounddevice soundfile
```

