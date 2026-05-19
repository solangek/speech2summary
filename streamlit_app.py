import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
UPLOAD_TYPES = ["wav", "mp3", "m4a", "m4v", "ogg", "flac", "webm"]
VIDEO_EXTENSIONS = {".m4v"}
UI_LANGUAGE_OPTIONS = ["fr", "en"]
TRANSCRIPTION_LANGUAGES = {
    "auto": None,
    "fr": "fr",
    "en": "en",
    "he": "he",
}
SUMMARY_OUTPUT_LANGUAGES = {
    "fr": "French (Français)",
    "en": "English",
    "he": "Hebrew",
    "ru" : "Russian"
}

TRANSLATIONS = {
    "fr": {
        "page_title": "Speech2Summary",
        "app_title": "Mon Assistant Perso",
        "app_caption": "Enregistrez ou téléversez un fichier audio → transcription → création d'un résumé structuré",
        "settings_header": "Paramètres",
        "ui_language_label": "Langue de l'interface",
        "transcription_language_label": "Langue de l'enregistrement",
        "transcription_language_help": "Aide le modèle de transcription en indiquant la langue parlée principale.",
        "summary_language_label": "Langue du résumé",
        "summary_language_help": "Force la langue de sortie du résumé, indépendamment de la langue de la transcription.",
        "auto_summarize": "Résumer automatiquement",
        "include_transcript": "Inclure la transcription dans le fichier",
        "compress_audio": "Compresser l'audio avant transcription",
        "compress_audio_help": "Réencode en Opus mono 16 kHz (≈ 16 kbps) via ffmpeg pour les fichiers ≥ 5 Mo. Réduit fortement la taille sans perte de précision pour Whisper.",
        "drive_connected": "Google Drive connecté",
        "disconnect_drive": "Déconnecter Drive",
        "connect_drive": "Connecter Google Drive",
        "tab_record": "Enregistrer",
        "tab_upload": "Envoyer un fichier audio",
        "recording_unavailable": "Enregistrement non disponible — mettez Streamlit à jour : `pip install --upgrade streamlit` (version 1.31+ requise)",
        "recording_input_label": "Cliquez sur le micro ci-dessous pour démarrer, cliquez à nouveau pour arrêter",
        "upload_audio_label": "Téléverser un fichier audio",
        "new_button": "🔄 Nouveau",
        "run_button": "Transcrire & Résumer",
        "extract_audio_spinner": "Extraction de la piste audio du fichier vidéo…",
        "extract_audio_caption": "Piste audio extraite ({size_mb:.2f} Mo)",
        "extract_audio_ffmpeg_error": "Impossible de traiter le fichier vidéo : ffmpeg est introuvable.",
        "extract_audio_failed_error": "Impossible d'extraire l'audio du fichier vidéo : {error}",
        "compress_audio_spinner": "Compression de l'audio ({size_mb:.1f} Mo)…",
        "compress_audio_caption": "Audio compressé à {size_mb:.2f} Mo",
        "compress_audio_ffmpeg_warning": "ffmpeg introuvable — envoi de l'audio non compressé.",
        "compress_audio_failed_warning": "Compression échouée, envoi du fichier original : {error}",
        "transcription_progress": "Transcription en cours…",
        "transcription_done": "Transcription terminée",
        "transcription_failed": "La transcription a échoué : {error}",
        "no_speech_detected": "Pas de conversation détectée dans l'audio.",
        "summary_progress": "Résumé en cours…",
        "summary_done": "Résumé terminé",
        "summary_quota_error": "Le modèle de résumé est temporairement indisponible (limite de quota atteinte, 20 requêtes/jour). Veuillez réessayer plus tard.",
        "summary_unavailable_error": "Gemini est temporairement surchargé (503). Réessayez dans quelques instants.",
        "summary_failed": "Échec du résumé : {error}",
        "summary_header": "Résumé",
        "full_transcript": "Transcript complet",
        "save_drive_spinner": "Sauvegarde sur Google Drive…",
        "save_drive_success": "Sauvegardé sur Google Drive.",
        "save_drive_failed": "Sauvegarde Drive échouée : {error}",
        "download_summary": "Télécharger le résumé",
        "history_header": "Historique",
        "history_empty": "Aucun résumé enregistré.",
        "download_file": "Télécharger",
        "generic_error": "Erreur",
        "history_failed": "Impossible de charger l'historique : {error}",
        "language_auto": "Détection automatique",
        "language_fr": "Français",
        "language_en": "Anglais",
        "language_he": "Hébreu (עברית)",
        "language_ru": "Russe (Русский)",
        "ui_language_fr": "Français",
        "ui_language_en": "English",
    },
    "en": {
        "page_title": "Speech2Summary",
        "app_title": "My Personal Assistant",
        "app_caption": "Record or upload an audio file → transcription → structured summary generation",
        "settings_header": "Settings",
        "ui_language_label": "Interface language",
        "transcription_language_label": "Recording language",
        "transcription_language_help": "Helps the transcription model by indicating the main spoken language.",
        "summary_language_label": "Summary language",
        "summary_language_help": "Forces the summary output language independently from the transcription language.",
        "auto_summarize": "Summarize automatically",
        "include_transcript": "Include transcript in the file",
        "compress_audio": "Compress audio before transcription",
        "compress_audio_help": "Re-encodes to mono 16 kHz Opus (≈ 16 kbps) via ffmpeg for files ≥ 5 MB. Greatly reduces size without hurting Whisper accuracy.",
        "drive_connected": "Google Drive connected",
        "disconnect_drive": "Disconnect Drive",
        "connect_drive": "Connect Google Drive",
        "tab_record": "Record",
        "tab_upload": "Upload audio file",
        "recording_unavailable": "Recording not available — update Streamlit: `pip install --upgrade streamlit` (version 1.31+ required)",
        "recording_input_label": "Click the microphone below to start, then click again to stop",
        "upload_audio_label": "Upload an audio file",
        "new_button": "🔄 New",
        "run_button": "Transcribe & Summarize",
        "extract_audio_spinner": "Extracting audio track from the video file…",
        "extract_audio_caption": "Audio track extracted ({size_mb:.2f} MB)",
        "extract_audio_ffmpeg_error": "Cannot process the video file: ffmpeg was not found.",
        "extract_audio_failed_error": "Unable to extract audio from the video file: {error}",
        "compress_audio_spinner": "Compressing audio ({size_mb:.1f} MB)…",
        "compress_audio_caption": "Audio compressed to {size_mb:.2f} MB",
        "compress_audio_ffmpeg_warning": "ffmpeg not found — sending uncompressed audio.",
        "compress_audio_failed_warning": "Compression failed, sending the original file instead: {error}",
        "transcription_progress": "Transcribing…",
        "transcription_done": "Transcription complete",
        "transcription_failed": "Transcription failed: {error}",
        "no_speech_detected": "No conversation detected in the audio.",
        "summary_progress": "Summarizing…",
        "summary_done": "Summary complete",
        "summary_quota_error": "The summary model is temporarily unavailable (quota limit reached, 20 requests/day). Please try again later.",
        "summary_unavailable_error": "Gemini is temporarily overloaded (503). Please try again in a moment.",
        "summary_failed": "Summary failed: {error}",
        "summary_header": "Summary",
        "full_transcript": "Full transcript",
        "save_drive_spinner": "Saving to Google Drive…",
        "save_drive_success": "Saved to Google Drive.",
        "save_drive_failed": "Drive save failed: {error}",
        "download_summary": "Download summary",
        "history_header": "History",
        "history_empty": "No saved summaries.",
        "download_file": "Download",
        "generic_error": "Error",
        "history_failed": "Unable to load history: {error}",
        "language_auto": "Auto-detect",
        "language_fr": "French",
        "language_en": "English",
        "language_he": "Hebrew (עברית)",
        "language_ru": "Russian (Русский)",
        "ui_language_fr": "Français",
        "ui_language_en": "English",
    },
}

DEFAULT_PROMPT = """You are a meeting/lecture assistant. Analyze the following transcript and provide a structured summary.

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


def get_ui_language() -> str:
    return st.session_state.get("ui_language", "fr")


def tr(key: str, **kwargs) -> str:
    language = get_ui_language()
    template = TRANSLATIONS.get(language, TRANSLATIONS["fr"]).get(key, key)
    return template.format(**kwargs)


def get_language_label(language_key: str, context: str = "transcription") -> str:
    if context == "transcription" and language_key == "auto":
        return tr("language_auto")
    return tr(f"language_{language_key}")


def get_ui_language_label(language_code: str) -> str:
    return tr(f"ui_language_{language_code}")


def build_summary_prompt(transcript: str, input_language: str, output_language: str) -> str:
    if input_language == "auto-detected from the transcript":
        transcript_language_instruction = "- The transcript language was auto-detected."
    else:
        transcript_language_instruction = f"- The transcript language is {input_language}."

    return (
        f"{get_default_prompt()}\n\n"
        f"Important:\n"
        f"{transcript_language_instruction}\n"
        f"- Write the final summary in {output_language}.\n"
        f"- Do not translate or alter the source transcript itself.\n\n"
        f"Transcript:\n{transcript}"
    )


def get_summary_input_language_label(transcription_language_key: str) -> str:
    if transcription_language_key == "auto":
        return "auto-detected from the transcript"
    return SUMMARY_OUTPUT_LANGUAGES[transcription_language_key]


def get_gemini_keys() -> list[str]:
    """Return list of Gemini API keys. Handles both single string and array."""
    raw = st.secrets.get("GEMINI_API_KEY", [])
    if isinstance(raw, str):
        return [raw]
    return list(raw)


def get_current_gemini_key() -> str:
    """Get the current active Gemini key from global rotation state."""
    keys = get_gemini_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY configured in secrets")
    if "gemini_key_index" not in st.session_state:
        st.session_state.gemini_key_index = 0
    idx = st.session_state.gemini_key_index
    return keys[idx % len(keys)]


def rotate_to_next_gemini_key() -> None:
    """Switch to the next Gemini key globally after current key fails."""
    keys = get_gemini_keys()
    if "gemini_key_index" not in st.session_state:
        st.session_state.gemini_key_index = 0
    st.session_state.gemini_key_index = (st.session_state.gemini_key_index + 1) % len(keys)


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


def transcode_audio_for_transcription(audio_bytes: bytes, input_filename: str) -> tuple[bytes, str]:
    input_suffix = os.path.splitext(input_filename)[1] or ".bin"
    output_name = f"{os.path.splitext(os.path.basename(input_filename))[0] or 'recording'}.ogg"
    input_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as input_tmp:
            input_tmp.write(audio_bytes)
            input_path = input_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as output_tmp:
            output_path = output_tmp.name

        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", input_path,
                "-vn",
                "-ac", "1", "-ar", "16000",
                "-c:a", "libopus", "-b:a", "16k",
                output_path,
            ],
            capture_output=True,
            check=True,
        )

        with open(output_path, "rb") as transcoded:
            return transcoded.read(), output_name
    finally:
        for path in (input_path, output_path):
            if path and Path(path).exists():
                Path(path).unlink()


def compress_audio(audio_bytes: bytes, input_filename: str) -> tuple[bytes, str]:
    return transcode_audio_for_transcription(audio_bytes, input_filename)


def is_video_upload(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS


def transcribe_with_retry(audio_bytes: bytes, filename: str, model: str, language: str | None, max_attempts: int = 4) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    for attempt in range(max_attempts):
        try:
            request_kwargs = {
                "file": (filename, audio_bytes),
                "model": model,
            }
            if language:
                request_kwargs["language"] = language

            result = client.audio.transcriptions.create(
                **request_kwargs,
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


def run_with_keepalive(fn, *args, progress, estimated_seconds: float, label: str, done_label: str):
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        while not future.done():
            elapsed = time.monotonic() - started
            ratio = min(0.95, elapsed / estimated_seconds)
            progress.progress(ratio, text=f"{label} {int(elapsed)}s")
            time.sleep(1.0)
        progress.progress(1.0, text=done_label)
        return future.result()


def transcribe_with_keepalive(audio_bytes: bytes, filename: str, model: str, language: str | None, progress) -> str:
    estimated = max(15.0, len(audio_bytes) / 1_500_000 * 15)
    return run_with_keepalive(
        transcribe_with_retry, audio_bytes, filename, model, language,
        progress=progress, estimated_seconds=estimated,
        label=tr("transcription_progress"), done_label=tr("transcription_done"),
    )


def transcribe(audio_bytes: bytes, filename: str, model: str, language: str | None) -> str:
    return transcribe_with_retry(audio_bytes, filename, model, language)


def summarize_with_retry(transcript: str, model: str, input_language: str, output_language: str, max_attempts: int | None = None) -> str:
    keys = get_gemini_keys()
    if max_attempts is None:
        max_attempts = len(keys)

    prompt = build_summary_prompt(transcript, input_language, output_language)
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        # Use the current global key
        key = get_current_gemini_key()
        client = genai.Client(api_key=key)

        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text or ""
        except Exception as e:
            last_error = e
            msg = str(e)

            # Check if error is transient or quota-related
            transient = "500" in msg or "deadline" in msg.lower()
            quota_error = "RESOURCE_EXHAUSTED" in msg or "429" in msg or "UNAVAILABLE" in msg or "503" in msg

            # If quota error, rotate to next key globally and retry
            if quota_error and attempt < max_attempts - 1:
                rotate_to_next_gemini_key()
                continue
            # If transient, retry with backoff on same key
            elif transient and attempt < max_attempts - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise

    if last_error:
        raise last_error
    return ""


def summarize_with_keepalive(transcript: str, model: str, input_language: str, output_language: str, progress) -> str:
    estimated = max(8.0, len(transcript) / 6000 * 2)
    return run_with_keepalive(
        summarize_with_retry, transcript, model, input_language, output_language,
        progress=progress, estimated_seconds=estimated,
        label=tr("summary_progress"), done_label=tr("summary_done"),
    )


def summarize(transcript: str, model: str, input_language: str, output_language: str) -> str:
    return summarize_with_retry(transcript, model, input_language, output_language)


def build_download_text(transcript: str, summary: str, with_transcript: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"Date: {now}\n\n## RÉSUMÉ\n\n{summary}\n"
    if with_transcript:
        text += f"\n---\n\n## TRANSCRIPT\n\n{transcript}\n"
    return text


def build_processing_key(
    audio_bytes: bytes,
    audio_filename: str,
    transcription_language_key: str,
    summary_language_key: str,
    compress_enabled: bool,
    groq_model: str,
    gemini_model: str,
) -> str:
    digest = hashlib.sha256(audio_bytes)
    for part in (
        audio_filename,
        transcription_language_key,
        summary_language_key,
        str(compress_enabled),
        groq_model,
        gemini_model,
    ):
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


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

st.set_page_config(page_title=tr("page_title"), page_icon="🎙️", layout="centered")
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title(tr("app_title"))
    st.caption(tr("app_caption"))
with col_btn:
    st.write("")  # alignement vertical
    btn_nouveau = st.empty()

if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0

rk = st.session_state.reset_key

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header(tr("settings_header"))
    st.selectbox(
        tr("ui_language_label"),
        options=UI_LANGUAGE_OPTIONS,
        format_func=get_ui_language_label,
        key="ui_language",
    )
    groq_model = GROQ_MODELS[0]
    transcription_language_key = st.selectbox(
        tr("transcription_language_label"),
        options=list(TRANSCRIPTION_LANGUAGES.keys()),
        index=0,
        format_func=lambda key: get_language_label(key, context="transcription"),
        help=tr("transcription_language_help"),
    )
    transcription_language = TRANSCRIPTION_LANGUAGES.get(transcription_language_key, None)
    gemini_model = GEMINI_MODELS[0]
    summary_language_key = st.selectbox(
        tr("summary_language_label"),
        options=list(SUMMARY_OUTPUT_LANGUAGES.keys()),
        index=0,
        format_func=lambda key: get_language_label(key, context="summary"),
        help=tr("summary_language_help"),
    )
    summary_output_language = SUMMARY_OUTPUT_LANGUAGES.get(summary_language_key, "French")
    auto_transcribe = st.checkbox(tr("auto_summarize"), value=True)
    include_transcript = st.checkbox(tr("include_transcript"), value=False)
    compress_enabled = st.checkbox(
        tr("compress_audio"),
        value=True,
        help=tr("compress_audio_help"),
    )
    drive_configured = all(k in st.secrets for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REDIRECT_URI"))
    if drive_configured:
        st.divider()
        if "credentials" in st.session_state:
            st.success(tr("drive_connected"))
            if st.button(tr("disconnect_drive")):
                del st.session_state["credentials"]
                st.rerun()
        else:
            flow = make_flow()
            auth_url, state = flow.authorization_url(prompt="consent", access_type="offline")
            st.session_state.oauth_state = state
            st.link_button(tr("connect_drive"), auth_url)

# ── Audio input ──────────────────────────────────────────────────────────────

tab_record, tab_upload = st.tabs([tr("tab_record"), tr("tab_upload")])

with tab_record:
    if not hasattr(st, "audio_input"):
        st.error(tr("recording_unavailable"))
    else:
        with st.container(border=True):
            audio_input = st.audio_input(tr("recording_input_label"), key=f"rec_{rk}")

        if audio_input:
            st.audio(audio_input)
            audio_bytes = audio_input.read()
            audio_filename = "recording.wav"

with tab_upload:
    uploaded = st.file_uploader(tr("upload_audio_label"), type=UPLOAD_TYPES, key=f"upl_{rk}")
    if uploaded:
        audio_filename = uploaded.name
        audio_bytes = uploaded.read()
        if is_video_upload(audio_filename):
            st.video(audio_bytes)
        else:
            st.audio(audio_bytes)

audio_ready = "audio_bytes" in dir() and audio_bytes

processing_key = None
result_cache: dict[str, dict[str, str]] = st.session_state.setdefault("result_cache", {})
cached_result: dict[str, str] | None = None
if audio_ready:
    processing_key = build_processing_key(
        audio_bytes,
        audio_filename,
        transcription_language_key,
        summary_language_key,
        compress_enabled,
        groq_model,
        gemini_model,
    )
    cached_result = result_cache.get(processing_key)

if audio_ready:
    btn_nouveau.button(tr("new_button"), on_click=reset, use_container_width=True)

if not auto_transcribe:
    run = st.button(tr("run_button"), type="primary", disabled=not audio_ready)
else:
    run = bool(audio_ready and cached_result is None)

# ── Transcription & summary ──────────────────────────────────────────────────

if run and audio_ready:
    assert processing_key is not None
    original_size = len(audio_bytes)
    requires_audio_extraction = is_video_upload(audio_filename)
    summary_input_language = get_summary_input_language_label(transcription_language_key)
    log_event("run_started", source=audio_filename, bytes=original_size, drive_connected="credentials" in st.session_state)

    if requires_audio_extraction:
        with st.spinner(tr("extract_audio_spinner")):
            t0 = time.monotonic()
            try:
                audio_bytes, audio_filename = transcode_audio_for_transcription(audio_bytes, audio_filename)
                st.caption(tr("extract_audio_caption", size_mb=len(audio_bytes) / 1_000_000))
                log_event("extract_audio_done", before_bytes=original_size, after_bytes=len(audio_bytes), duration_s=round(time.monotonic() - t0, 2))
            except FileNotFoundError:
                log_event("extract_audio_failed", reason="ffmpeg_not_found")
                st.error(tr("extract_audio_ffmpeg_error"))
                st.stop()
            except subprocess.CalledProcessError as e:
                log_event("extract_audio_failed", error=e.stderr.decode(errors="ignore")[:200])
                st.error(tr("extract_audio_failed_error", error=e.stderr.decode(errors="ignore")[:200]))
                st.stop()

    if compress_enabled and not requires_audio_extraction and len(audio_bytes) >= COMPRESS_THRESHOLD_BYTES:
        with st.spinner(tr("compress_audio_spinner", size_mb=len(audio_bytes) / 1_000_000)):
            t0 = time.monotonic()
            try:
                audio_bytes, audio_filename = compress_audio(audio_bytes, audio_filename)
                st.caption(tr("compress_audio_caption", size_mb=len(audio_bytes) / 1_000_000))
                log_event("compress_done", before_bytes=original_size, after_bytes=len(audio_bytes), duration_s=round(time.monotonic() - t0, 2))
            except FileNotFoundError:
                st.warning(tr("compress_audio_ffmpeg_warning"))
                log_event("compress_skipped", reason="ffmpeg_not_found")
            except subprocess.CalledProcessError as e:
                st.warning(tr("compress_audio_failed_warning", error=e.stderr.decode(errors="ignore")[:200]))
                log_event("compress_failed", error=e.stderr.decode(errors="ignore")[:200])

    progress_bar = st.progress(0.0, text=tr("transcription_progress"))
    t0 = time.monotonic()
    try:
        transcript = transcribe_with_keepalive(audio_bytes, audio_filename, groq_model, transcription_language, progress_bar)
    except Exception as e:
        progress_bar.empty()
        log_event("transcribe_failed", model=groq_model, language=transcription_language_key, bytes=len(audio_bytes), duration_s=round(time.monotonic() - t0, 2), error=str(e)[:300])
        st.error(tr("transcription_failed", error=e))
        st.stop()
    progress_bar.empty()
    log_event("transcribe_done", model=groq_model, language=transcription_language_key, bytes=len(audio_bytes), duration_s=round(time.monotonic() - t0, 2), transcript_chars=len(transcript))

    if not transcript:
        st.warning(tr("no_speech_detected"))
        log_event("transcribe_empty")
        st.stop()

    summary_progress = st.progress(0.0, text=tr("summary_progress"))
    t0 = time.monotonic()
    try:
        summary = summarize_with_keepalive(transcript, gemini_model, summary_input_language, summary_output_language, summary_progress)
        log_event("summarize_done", model=gemini_model, input_language=summary_input_language, output_language=summary_output_language, transcript_chars=len(transcript), summary_chars=len(summary or ""), duration_s=round(time.monotonic() - t0, 2))
    except Exception as e:
        summary_progress.empty()
        log_event("summarize_failed", model=gemini_model, input_language=summary_input_language, output_language=summary_output_language, error=str(e)[:300], duration_s=round(time.monotonic() - t0, 2))
        err = str(e)
        if "RESOURCE_EXHAUSTED" in err:
            st.error(tr("summary_quota_error"))
        elif "UNAVAILABLE" in err or "503" in err:
            st.error(tr("summary_unavailable_error"))
        else:
            st.error(tr("summary_failed", error=e))
        st.stop()
    summary_progress.empty()

    result_cache[processing_key] = {
        "transcript": transcript,
        "summary": summary,
        "filename": f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    }
    cached_result = result_cache[processing_key]

    content = build_download_text(transcript, summary, include_transcript)
    filename = cached_result["filename"]

    if "credentials" in st.session_state:
        with st.spinner(tr("save_drive_spinner")):
            try:
                creds = Credentials(**st.session_state.credentials)
                save_to_drive(creds, content, filename)
                st.success(tr("save_drive_success"))
                log_event("drive_saved", filename=filename, bytes=len(content))
            except Exception as e:
                st.warning(tr("save_drive_failed", error=e))
                log_event("drive_save_failed", error=str(e)[:300])


if audio_ready and cached_result:
    transcript = cached_result["transcript"]
    summary = cached_result["summary"]
    filename = cached_result["filename"]
    content = build_download_text(transcript, summary, include_transcript)

    st.subheader(tr("summary_header"))
    st.markdown(summary)

    with st.expander(tr("full_transcript")):
        st.write(transcript)

    st.download_button(
        label=tr("download_summary"),
        data=content,
        file_name=filename,
        mime="text/plain",
    )

# ── Historique ───────────────────────────────────────────────────────────────

if "credentials" in st.session_state:
    st.divider()
    st.subheader(tr("history_header"))
    try:
        creds = Credentials(**st.session_state.credentials)
        files = list_drive_files(creds)
        if not files:
            st.caption(tr("history_empty"))
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
                            label=tr("download_file"),
                            data=data,
                            file_name=f["name"],
                            mime="text/plain",
                            key=f["id"],
                        )
                    except Exception:
                        st.caption(tr("generic_error"))
    except Exception as e:
        st.caption(tr("history_failed", error=e))
