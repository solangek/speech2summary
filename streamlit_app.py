import os
import tempfile
from datetime import datetime

import streamlit as st
from google import genai
from groq import Groq

GROQ_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

DEFAULT_PROMPT = """You are a meeting/lecture assistant. Analyze the following transcript and provide a structured summary.
Respond in the same language as the transcript.

Format your response EXACTLY as:

## Key Elements
- [point 1]
- [point 2]

## Actions Items
- [item 1] (or "None identified" if there are none)

## Decisions Made
- [decision 1] (or "None identified" if there are none)"""


def get_default_prompt() -> str:
    return st.secrets.get("SUMMARIZATION_PROMPT", DEFAULT_PROMPT)


def transcribe(audio_bytes: bytes, filename: str, model: str) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    result = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=model,
    )
    return result.text.strip()


def summarize(transcript: str, model: str) -> str:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    prompt = f"{get_default_prompt()}\n\nTranscript:\n{transcript}"
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def build_download_text(transcript: str, summary: str, with_transcript: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"Date: {now}\n\n## RÉSUMÉ\n\n{summary}\n"
    if with_transcript:
        text += f"\n---\n\n## TRANSCRIPT\n\n{transcript}\n"
    return text


def reset():
    st.session_state.reset_key = st.session_state.get("reset_key", 0) + 1


st.set_page_config(page_title="Speech2Summary", page_icon="🎙️", layout="centered")
st.title("Mon Assistant Perso")
st.caption("Enregistrez ou téléversez un fichier audio → transcription → création d'un resumé structuré")

if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0

rk = st.session_state.reset_key

with st.sidebar:
    st.header("Settings")
    groq_model = st.selectbox("Transcription model (Groq)", GROQ_MODELS)
    gemini_model = st.selectbox("Summarization model (Gemini)", GEMINI_MODELS)
    auto_transcribe = st.checkbox("Résumer automatiquement", value=True)
    include_transcript = st.checkbox("Inclure la transcription dans le fichier", value=False)

tab_record, tab_upload = st.tabs(["Enregistrer", "Envoyer un fichier audio"])

with tab_record:
    audio_input = st.audio_input("Cliquez pour enregistrer depuis votre micro", key=f"rec_{rk}")
    if audio_input:
        st.audio(audio_input)
        audio_bytes = audio_input.read()
        audio_filename = "recording.wav"

with tab_upload:
    uploaded = st.file_uploader("Téléverser un fichier audio", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"], key=f"upl_{rk}")
    if uploaded:
        st.audio(uploaded)
        audio_bytes = uploaded.read()
        audio_filename = uploaded.name

audio_ready = "audio_bytes" in dir() and audio_bytes

if not auto_transcribe:
    run = st.button("Transcrire & Résumer", type="primary", disabled=not audio_ready)
else:
    run = bool(audio_ready)

if run and audio_ready:
        with st.spinner("Transcription en cours…"):
            try:
                transcript = transcribe(audio_bytes, audio_filename, groq_model)
            except Exception as e:
                st.error(f"La transcription echoué: {e}")
                st.stop()

        if not transcript:
            st.warning("Pas de conversation détectée dans l'audio.")
            st.stop()

        with st.spinner("Resumé en cours…"):
            try:
                summary = summarize(transcript, gemini_model)
            except Exception as e:
                st.error(f"Echec fu résumé: {e}")
                st.stop()

        st.subheader("Résumé")
        st.markdown(summary)

        with st.expander("Transcript complet"):
            st.write(transcript)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Télécharger le resumé",
                data=build_download_text(transcript, summary, include_transcript),
                file_name=f"summary_{timestamp}.txt",
                mime="text/plain",
            )
        with col2:
            st.button("Nouvel enregistrement", on_click=reset)
