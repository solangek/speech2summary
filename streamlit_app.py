import json
import logging
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import streamlit as st
from google import genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from groq import Groq, RateLimitError

GROQ_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

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


_usage_logger = logging.getLogger("speech2summary.usage")
if not _usage_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _usage_logger.addHandler(_handler)
    _usage_logger.setLevel(logging.INFO)
    _usage_logger.propagate = False


def log_event(event: str, **fields) -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "session_id": st.session_state.session_id,
        **fields,
    }
    _usage_logger.info(json.dumps(payload, ensure_ascii=False, default=str))


COMPRESS_THRESHOLD_BYTES = 5_000_000


def compress_audio(audio_bytes: bytes) -> tuple[bytes, str]:
    result = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
            "-ac", "1", "-ar", "16000",
            "-c:a", "libopus", "-b:a", "16k",
            "-f", "ogg", "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    return result.stdout, "recording.ogg"


def transcribe_with_retry(audio_bytes: bytes, filename: str, model: str, max_attempts: int = 4) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    for attempt in range(max_attempts):
        try:
            result = client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model=model,
            )
            return result.text.strip()
        except RateLimitError as e:
            if attempt == max_attempts - 1:
                raise
            retry_after = 2 ** attempt
            try:
                retry_after = int(e.response.headers.get("retry-after", retry_after))
            except (AttributeError, ValueError, TypeError):
                pass
            time.sleep(retry_after)
    return ""


def transcribe_with_keepalive(audio_bytes: bytes, filename: str, model: str, progress) -> str:
    estimated_seconds = max(15.0, len(audio_bytes) / 1_500_000 * 15)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(transcribe_with_retry, audio_bytes, filename, model)
        while not future.done():
            elapsed = time.monotonic() - started
            ratio = min(0.95, elapsed / estimated_seconds)
            progress.progress(ratio, text=f"Transcription en cours… {int(elapsed)}s")
            time.sleep(1.0)
        progress.progress(1.0, text="Transcription terminée")
        return future.result()


def transcribe(audio_bytes: bytes, filename: str, model: str) -> str:
    return transcribe_with_retry(audio_bytes, filename, model)


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


def make_flow() -> Flow:
    client_config = {
        "web": {
            "client_id": st.secrets["GOOGLE_CLIENT_ID"],
            "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [st.secrets["GOOGLE_REDIRECT_URI"]],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=SCOPES)
    flow.redirect_uri = st.secrets["GOOGLE_REDIRECT_URI"]
    return flow


def drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds)


DRIVE_FOLDER_NAME = "Speech2Summary"


def get_or_create_folder(service) -> str:
    results = service.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and name='{DRIVE_FOLDER_NAME}' and trashed=false",
        fields="files(id)",
    ).execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    folder = service.files().create(
        body={"name": DRIVE_FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"},
        fields="id",
    ).execute()
    return folder["id"]


def save_to_drive(creds: Credentials, content: str, filename: str) -> None:
    service = drive_service(creds)
    folder_id = get_or_create_folder(service)
    media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/plain")
    service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media,
    ).execute()


def list_drive_files(creds: Credentials) -> list[dict]:
    service = drive_service(creds)
    folder_id = get_or_create_folder(service)
    result = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, createdTime)",
        orderBy="createdTime desc",
        pageSize=50,
    ).execute()
    return result.get("files", [])


def download_drive_file(creds: Credentials, file_id: str) -> bytes:
    return drive_service(creds).files().get_media(fileId=file_id).execute()


def reset():
    st.session_state.reset_key = st.session_state.get("reset_key", 0) + 1


# ── OAuth callback handling ──────────────────────────────────────────────────

params = st.query_params
if "code" in params and "credentials" not in st.session_state:
    flow = make_flow()
    flow.fetch_token(code=params["code"])
    creds = flow.credentials
    st.session_state.credentials = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
    }
    st.query_params.clear()
    st.rerun()

# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Speech2Summary", page_icon="🎙️", layout="centered")
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title("Mon Assistant Perso")
    st.caption("Enregistrez ou téléversez un fichier audio → transcription → création d'un résumé structuré")
with col_btn:
    st.write("")  # alignement vertical
    btn_nouveau = st.empty()

if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0

rk = st.session_state.reset_key

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    groq_model = st.selectbox("Transcription model (Groq)", GROQ_MODELS)
    gemini_model = st.selectbox("Summarization model (Gemini)", GEMINI_MODELS)
    auto_transcribe = st.checkbox("Résumer automatiquement", value=True)
    include_transcript = st.checkbox("Inclure la transcription dans le fichier", value=False)
    compress_enabled = st.checkbox(
        "Compresser l'audio avant transcription",
        value=True,
        help="Réencode en Opus mono 16 kHz (≈ 16 kbps) via ffmpeg pour les fichiers ≥ 5 Mo. Réduit fortement la taille sans perte de précision pour Whisper.",
    )
    drive_configured = all(k in st.secrets for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REDIRECT_URI"))
    if drive_configured:
        st.divider()
        if "credentials" in st.session_state:
            st.success("Google Drive connecté")
            if st.button("Déconnecter Drive"):
                del st.session_state["credentials"]
                st.rerun()
        else:
            flow = make_flow()
            auth_url, state = flow.authorization_url(prompt="consent", access_type="offline")
            st.session_state.oauth_state = state
            st.link_button("Connecter Google Drive", auth_url)

# ── Audio input ──────────────────────────────────────────────────────────────

tab_record, tab_upload = st.tabs(["Enregistrer", "Envoyer un fichier audio"])

with tab_record:
    if not hasattr(st, "audio_input"):
        st.error("Enregistrement non disponible — mettez Streamlit à jour : `pip install --upgrade streamlit` (version 1.31+ requise)")
    else:
        with st.container(border=True):
            audio_input = st.audio_input("Cliquez sur le micro ci-dessous pour démarrer, cliquez à nouveau pour arrêter", key=f"rec_{rk}")

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

if audio_ready:
    btn_nouveau.button("🔄 Nouveau", on_click=reset, use_container_width=True)

if not auto_transcribe:
    run = st.button("Transcrire & Résumer", type="primary", disabled=not audio_ready)
else:
    run = bool(audio_ready)

# ── Transcription & summary ──────────────────────────────────────────────────

if run and audio_ready:
    original_size = len(audio_bytes)
    log_event("run_started", source=audio_filename, bytes=original_size, drive_connected="credentials" in st.session_state)

    if compress_enabled and len(audio_bytes) >= COMPRESS_THRESHOLD_BYTES:
        with st.spinner(f"Compression de l'audio ({len(audio_bytes) / 1_000_000:.1f} Mo)…"):
            t0 = time.monotonic()
            try:
                audio_bytes, audio_filename = compress_audio(audio_bytes)
                st.caption(f"Audio compressé à {len(audio_bytes) / 1_000_000:.2f} Mo")
                log_event("compress_done", before_bytes=original_size, after_bytes=len(audio_bytes), duration_s=round(time.monotonic() - t0, 2))
            except FileNotFoundError:
                st.warning("ffmpeg introuvable — envoi de l'audio non compressé.")
                log_event("compress_skipped", reason="ffmpeg_not_found")
            except subprocess.CalledProcessError as e:
                st.warning(f"Compression échouée, envoi du fichier original : {e.stderr.decode(errors='ignore')[:200]}")
                log_event("compress_failed", error=e.stderr.decode(errors="ignore")[:200])

    progress_bar = st.progress(0.0, text="Transcription en cours…")
    t0 = time.monotonic()
    try:
        transcript = transcribe_with_keepalive(audio_bytes, audio_filename, groq_model, progress_bar)
    except Exception as e:
        progress_bar.empty()
        log_event("transcribe_failed", model=groq_model, bytes=len(audio_bytes), duration_s=round(time.monotonic() - t0, 2), error=str(e)[:300])
        st.error(f"La transcription a échoué : {e}")
        st.stop()
    progress_bar.empty()
    log_event("transcribe_done", model=groq_model, bytes=len(audio_bytes), duration_s=round(time.monotonic() - t0, 2), transcript_chars=len(transcript))

    if not transcript:
        st.warning("Pas de conversation détectée dans l'audio.")
        log_event("transcribe_empty")
        st.stop()

    with st.spinner("Résumé en cours…"):
        t0 = time.monotonic()
        try:
            summary = summarize(transcript, gemini_model)
            log_event("summarize_done", model=gemini_model, transcript_chars=len(transcript), summary_chars=len(summary or ""), duration_s=round(time.monotonic() - t0, 2))
        except Exception as e:
            log_event("summarize_failed", model=gemini_model, error=str(e)[:300], duration_s=round(time.monotonic() - t0, 2))
            # check if error is 429 RESOURCE_EXHAUSTED from Gemini API and show a specific message
            if "RESOURCE_EXHAUSTED" in str(e):
                st.error("Le modèle de résumé est temporairement indisponible (limite de quota atteinte, 20 requêtes/jour). Veuillez réessayer plus tard.")
            else:
                st.error(f"Échec du résumé : {e}")
            st.stop()

    st.subheader("Résumé")
    st.markdown(summary)

    with st.expander("Transcript complet"):
        st.write(transcript)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content = build_download_text(transcript, summary, include_transcript)
    filename = f"summary_{timestamp}.txt"

    if "credentials" in st.session_state:
        with st.spinner("Sauvegarde sur Google Drive…"):
            try:
                creds = Credentials(**st.session_state.credentials)
                save_to_drive(creds, content, filename)
                st.success("Sauvegardé sur Google Drive.")
                log_event("drive_saved", filename=filename, bytes=len(content))
            except Exception as e:
                st.warning(f"Sauvegarde Drive échouée : {e}")
                log_event("drive_save_failed", error=str(e)[:300])

    st.download_button(
        label="Télécharger le résumé",
        data=content,
        file_name=filename,
        mime="text/plain",
    )

# ── Historique ───────────────────────────────────────────────────────────────

if "credentials" in st.session_state:
    st.divider()
    st.subheader("Historique")
    try:
        creds = Credentials(**st.session_state.credentials)
        files = list_drive_files(creds)
        if not files:
            st.caption("Aucun résumé enregistré.")
        else:
            for f in files:
                created = f["createdTime"][:10]
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{created} — {f['name']}")
                with col2:
                    try:
                        data = download_drive_file(creds, f["id"])
                        st.download_button(
                            label="Télécharger",
                            data=data,
                            file_name=f["name"],
                            mime="text/plain",
                            key=f["id"],
                        )
                    except Exception:
                        st.caption("Erreur")
    except Exception as e:
        st.caption(f"Impossible de charger l'historique : {e}")
