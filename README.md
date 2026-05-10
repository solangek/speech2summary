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

### API keys

The project uses Groq for transcription and Gemini for summarization.

For the CLI in `main.py`, set environment variables:

```bash
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"
```

`google-genai` also accepts `GOOGLE_API_KEY`; if both are set, it prefers `GOOGLE_API_KEY`.

For the Streamlit app, add the same values to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_key"
GEMINI_API_KEY = "your_gemini_key"
```

## Run the CLI

```bash
python3 main.py
```

Optional flags:
- `--output` — output directory for saved summaries (default: `~/summaries/`)
- `--groq-model` — transcription model (default: `whisper-large-v3-turbo`)
- `--gemini-model` — summarization model (default: `gemini-2.5-flash`)

Example:

```bash
python3 main.py --output ~/summaries --groq-model whisper-large-v3-turbo --gemini-model gemini-2.5-flash
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL in your browser, record audio or upload a file, and click **Transcribe & Summarize**.

## Note about CLI dependencies

`main.py` imports `numpy`, `sounddevice`, and `soundfile`, but they are not currently listed in `requirements.txt`.
If you want to use the CLI recorder, you may need to install them manually:

```bash
pip install numpy sounddevice soundfile
```

