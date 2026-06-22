"""Streamlit app: upload a vet X-ray (DICOM/JPG/PNG) → structured draft report.

Decision-support for a licensed veterinarian — the output is a DRAFT to review and
sign, never an autonomous diagnosis. Reuses shared plumbing from common.py.
"""
import time
from datetime import datetime

import streamlit as st
from google.oauth2.credentials import Credentials

import common
import xray

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
DRIVE_FOLDER = "VetXrayReports"
MAX_VIEWS = 6  # cap views per study to bound cost/latency

# ── OAuth callback handling ──────────────────────────────────────────────────────
params = st.query_params
if "code" in params and "credentials" not in st.session_state:
    flow = common.make_flow()
    flow.fetch_token(code=params["code"])
    creds = flow.credentials
    st.session_state.credentials = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or common.SCOPES),
    }
    st.query_params.clear()
    st.rerun()

# ── Page setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VetRadio", page_icon="🐾", layout="centered")
st.title("Assistant Radiologie Vétérinaire")
st.caption("Téléversez une ou plusieurs vues d'une même étude (DICOM / JPG / PNG) → brouillon de compte rendu structuré")
st.warning(xray.DISCLAIMER)

_, col_btn = st.columns([4, 1])
with col_btn:
    if st.button("🎧 Assistant Audio", use_container_width=True):
        st.switch_page("streamlit_app.py")

if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0
rk = st.session_state.reset_key

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Réglages")
    gemini_model = st.selectbox("Modèle d'interprétation (Gemini)", GEMINI_MODELS)
    st.caption("Flash : rapide et économique · Pro : raisonnement plus fin")

    if common.drive_configured():
        st.divider()
        if "credentials" in st.session_state:
            st.success("Google Drive connecté")
            if st.button("Déconnecter Drive"):
                del st.session_state["credentials"]
                st.rerun()
        else:
            flow = common.make_flow()
            auth_url, state = flow.authorization_url(prompt="consent", access_type="offline")
            st.session_state.oauth_state = state
            st.link_button("Connecter Google Drive", auth_url)

# ── Upload & preprocess ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Radiographie(s) — plusieurs vues de la même étude améliorent l'analyse",
    type=["dcm", "png", "jpg", "jpeg"], accept_multiple_files=True, key=f"upl_{rk}",
)

# Clinical context — always visible (improves analysis; DICOM fills blanks once loaded)
st.markdown("**Contexte clinique** — améliore l'analyse (optionnel mais recommandé)")
c1, c2 = st.columns(2)
species = c1.text_input("Espèce", placeholder="chien, chat, cheval…")
breed = c2.text_input("Race")
c3, c4, c5 = st.columns(3)
age = c3.text_input("Âge", placeholder="ex. 8 ans")
sex = c4.selectbox("Sexe", ["", "M", "M castré", "F", "F stérilisée"])
view = c5.text_input("Incidence / région", placeholder="thorax LAT…")
history = st.text_area(
    "Motif de l'examen et contexte clinique",
    placeholder="Motif, signes cliniques, antécédents, traitements en cours…",
    height=90,
)

if uploaded:
    if len(uploaded) > MAX_VIEWS:
        st.warning(f"Trop de fichiers — seules les {MAX_VIEWS} premières vues seront analysées.")
        uploaded = uploaded[:MAX_VIEWS]

    images, view_labels, names, total_bytes, study_meta = [], [], [], 0, {}
    for f in uploaded:
        fdata = f.read()
        total_bytes += len(fdata)
        try:
            png, m = xray.load_image(f.name, fdata)
        except Exception as e:
            common.log_event("preprocess_failed", source=f.name, error=str(e)[:300])
            st.error(f"Impossible de lire {f.name} : {e}")
            st.stop()
        images.append(png)
        names.append(f.name)
        view_labels.append(m.get("view", ""))
        for k in ("species", "breed", "body_part", "modality"):
            if m.get(k) and not study_meta.get(k):
                study_meta[k] = m[k]

    cols = st.columns(min(len(images), 3))
    for i, png in enumerate(images):
        cols[i % len(cols)].image(
            png, caption=view_labels[i] or names[i] or f"Vue {i + 1}", use_container_width=True
        )

    btn_label = "Interpréter" if len(images) == 1 else f"Interpréter les {len(images)} vues"
    if st.button(btn_label, type="primary"):
        meta = {
            "species": species or study_meta.get("species", ""),
            "breed": breed or study_meta.get("breed", ""),
            "age": age,
            "sex": sex,
            "view": view or (view_labels[0] if len(images) == 1 else ""),
            "body_part": study_meta.get("body_part", ""),
            "modality": study_meta.get("modality", ""),
            "history": history,
            "views": view_labels,
        }
        common.log_event(
            "run_started", source="; ".join(names), n_views=len(images), bytes=total_bytes,
            drive_connected="credentials" in st.session_state, model=gemini_model,
            has_context=bool(history.strip() or species or breed or age or sex or view),
        )
        progress = st.progress(0.0, text="Interprétation en cours…")
        t0 = time.monotonic()
        try:
            report = common.run_with_keepalive(
                xray.interpret_with_retry, images, gemini_model, meta,
                progress=progress, estimated_seconds=12.0 + 6.0 * len(images),
                label="Interprétation en cours…", done_label="Terminé",
            )
        except Exception as e:
            progress.empty()
            common.log_event("interpret_failed", model=gemini_model, error=str(e)[:300],
                             duration_s=round(time.monotonic() - t0, 2))
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err:
                st.error("Quota Gemini atteint. Réessayez plus tard.")
            elif "UNAVAILABLE" in err or "503" in err:
                st.error("Gemini est temporairement surchargé (503). Réessayez dans quelques instants.")
            else:
                st.error(f"Échec de l'interprétation : {e}")
            st.stop()
        progress.empty()
        common.log_event("interpret_done", model=gemini_model, n_views=len(images),
                         duration_s=round(time.monotonic() - t0, 2))

        report_md = xray.build_report_md(report)
        st.markdown(report_md)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compte_rendu_{timestamp}.md"
        ctx_summary = "; ".join(
            f"{lbl} : {meta.get(k)}"
            for k, lbl in (("species", "Espèce"), ("breed", "Race"), ("age", "Âge"),
                           ("sex", "Sexe"), ("view", "Incidence"))
            if meta.get(k)
        )
        views_summary = ", ".join(v for v in view_labels if v)
        header = f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nModèle : {gemini_model}\n"
        header += f"Vues : {len(images)}" + (f" ({views_summary})\n" if views_summary else "\n")
        if ctx_summary:
            header += f"Contexte : {ctx_summary}\n"
        if history.strip():
            header += f"Motif / antécédents : {history.strip()}\n"
        full = f"{header}\n{report_md}\n"

        if "credentials" in st.session_state:
            with st.spinner("Sauvegarde sur Google Drive…"):
                try:
                    creds = Credentials(**st.session_state.credentials)
                    common.save_text_to_drive(creds, full, filename, DRIVE_FOLDER)
                    for i, png in enumerate(images, 1):
                        common.save_bytes_to_drive(creds, png, f"radio_{timestamp}_vue{i}.png", "image/png", DRIVE_FOLDER)
                    st.success(f"Compte rendu et {len(images)} image(s) sauvegardés sur Google Drive.")
                    common.log_event("drive_saved", filename=filename, n_views=len(images))
                except Exception as e:
                    st.warning(f"Sauvegarde Drive échouée : {e}")
                    common.log_event("drive_save_failed", error=str(e)[:300])

        st.download_button("Télécharger le compte rendu", data=full, file_name=filename, mime="text/markdown")

# ── Historique ─────────────────────────────────────────────────────────────────────
if "credentials" in st.session_state:
    st.divider()
    st.subheader("Historique")
    try:
        creds = Credentials(**st.session_state.credentials)
        files = common.list_drive_files(creds, DRIVE_FOLDER)
        if not files:
            st.caption("Aucun compte rendu enregistré.")
        else:
            for f in files:
                created = f["createdTime"][:10]
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{created} — {f['name']}")
                with col2:
                    try:
                        content = common.download_drive_file(creds, f["id"])
                        st.download_button("Télécharger", data=content, file_name=f["name"], key=f["id"])
                    except Exception:
                        st.caption("Erreur")
    except Exception as e:
        st.caption(f"Impossible de charger l'historique : {e}")
