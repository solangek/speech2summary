"""Shared Streamlit helpers reused by the speech2summary and vet X-ray apps.

Kept separate so the X-ray app can reuse the stable plumbing (logging, keepalive,
Gemini key rotation, Google Drive) without modifying the live streamlit_app.py.
These mirror the helpers in streamlit_app.py; that file keeps its own copies for now.
"""
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# ── Logging (one JSON line per event, to stdout → Streamlit Cloud logs) ──────────
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


# ── Gemini API keys (rotation across multiple keys on quota errors) ──────────────
def get_gemini_keys() -> list[str]:
    """Return list of Gemini API keys. Handles both single string and array."""
    raw = st.secrets.get("GEMINI_API_KEY", [])
    if isinstance(raw, str):
        return [raw]
    return list(raw)


def get_current_gemini_key() -> str:
    keys = get_gemini_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY configured in secrets")
    if "gemini_key_index" not in st.session_state:
        st.session_state.gemini_key_index = 0
    return keys[st.session_state.gemini_key_index % len(keys)]


def rotate_to_next_gemini_key() -> None:
    keys = get_gemini_keys()
    if "gemini_key_index" not in st.session_state:
        st.session_state.gemini_key_index = 0
    st.session_state.gemini_key_index = (st.session_state.gemini_key_index + 1) % len(keys)


# ── Keepalive (run a blocking call while pinging a progress bar) ──────────────────
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


# ── Google Drive ─────────────────────────────────────────────────────────────────
def drive_configured() -> bool:
    return all(k in st.secrets for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REDIRECT_URI"))


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


def get_or_create_folder(service, folder_name: str) -> str:
    results = service.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false",
        fields="files(id)",
    ).execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    folder = service.files().create(
        body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"},
        fields="id",
    ).execute()
    return folder["id"]


def save_bytes_to_drive(creds: Credentials, data: bytes, filename: str, mimetype: str, folder_name: str) -> None:
    service = drive_service(creds)
    folder_id = get_or_create_folder(service, folder_name)
    media = MediaInMemoryUpload(data, mimetype=mimetype)
    service.files().create(body={"name": filename, "parents": [folder_id]}, media_body=media).execute()


def save_text_to_drive(creds: Credentials, content: str, filename: str, folder_name: str) -> None:
    save_bytes_to_drive(creds, content.encode("utf-8"), filename, "text/plain", folder_name)


def list_drive_files(creds: Credentials, folder_name: str) -> list[dict]:
    service = drive_service(creds)
    folder_id = get_or_create_folder(service, folder_name)
    result = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, createdTime)",
        orderBy="createdTime desc",
        pageSize=50,
    ).execute()
    return result.get("files", [])


def download_drive_file(creds: Credentials, file_id: str) -> bytes:
    return drive_service(creds).files().get_media(fileId=file_id).execute()
