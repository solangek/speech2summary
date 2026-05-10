import os
import tempfile
from datetime import datetime

import streamlit as st
from google import genai
from groq import Groq

GROQ_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]


def transcribe(audio_bytes: bytes, filename: str, model: str) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    result = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=model,
    )
    return result.text.strip()


def summarize(transcript: str, model: str) -> str:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    prompt = f"""You are a meeting/lecture assistant. Analyze the following transcript and provide a structured summary.
Respond in the same language as the transcript.

Format your response EXACTLY as:

## Key Points
- [point 1]
- [point 2]

## Action Items
- [item 1] (or "None identified" if there are none)

## Decisions Made
- [decision 1] (or "None identified" if there are none)

Transcript:
{transcript}"""

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def build_download_text(transcript: str, summary: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"Recording Date: {now}\n\n"
        "## SUMMARY\n\n"
        f"{summary}\n\n"
        "---\n\n"
        "## TRANSCRIPT\n\n"
        f"{transcript}\n"
    )


st.set_page_config(page_title="Speech2Summary", page_icon="🎙️", layout="centered")
st.title("Speech2Summary")
st.caption("Record or upload audio → transcribe with Groq → summarize with Gemini")

with st.sidebar:
    st.header("Settings")
    groq_model = st.selectbox("Transcription model (Groq)", GROQ_MODELS)
    gemini_model = st.selectbox("Summarization model (Gemini)", GEMINI_MODELS)

tab_record, tab_upload = st.tabs(["Record", "Upload file"])

with tab_record:
    audio_input = st.audio_input("Click to record from your microphone")
    if audio_input:
        st.audio(audio_input)
        audio_bytes = audio_input.read()
        audio_filename = "recording.wav"

with tab_upload:
    uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
    if uploaded:
        st.audio(uploaded)
        audio_bytes = uploaded.read()
        audio_filename = uploaded.name

audio_ready = "audio_bytes" in dir() and audio_bytes

if audio_ready:
    if st.button("Transcribe & Summarize", type="primary"):
        with st.spinner("Transcribing…"):
            try:
                transcript = transcribe(audio_bytes, audio_filename, groq_model)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

        if not transcript:
            st.warning("No speech detected in the audio.")
            st.stop()

        with st.spinner("Summarizing…"):
            try:
                summary = summarize(transcript, gemini_model)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                st.stop()

        st.subheader("Summary")
        st.markdown(summary)

        with st.expander("Full transcript"):
            st.write(transcript)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download summary + transcript",
            data=build_download_text(transcript, summary),
            file_name=f"summary_{timestamp}.txt",
            mime="text/plain",
        )
