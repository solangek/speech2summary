"""Vet X-ray interpretation with Gemini — global image understanding + report writing.

Gemini handles both perception (reading the whole radiograph) and natural-language
report drafting. Pure imaging (DICOM decode/windowing) lives in imaging.py.
"""
import io
import json
import time

from PIL import Image
import streamlit as st
from google import genai
from google.genai import types

import common
from imaging import load_image  # noqa: F401 (re-exported for the app)

DEFAULT_RADIOLOGY_PROMPT = """Tu es un assistant en radiologie vétérinaire. Tu rédiges un BROUILLON de compte rendu destiné à être relu et validé par un vétérinaire qualifié. Tu n'établis JAMAIS de diagnostic définitif.

Analyse la radiographie de façon systématique et structurée. Règles :
- Examine systématiquement toutes les régions anatomiques visibles (silhouette cardiaque, champs pulmonaires, plèvre, médiastin, abdomen, structures osseuses et tissus mous, selon la projection) avant de conclure.
- Décris uniquement ce qui est visible ; n'invente pas de structures ni de pathologies.
- Si la qualité d'image (exposition, positionnement, mouvement) limite l'interprétation, dis-le explicitement.
- Hiérarchise les diagnostics différentiels du plus au moins probable.
- N'exprime aucune certitude excessive ; recommande la corrélation avec les signes cliniques et, au besoin, des examens complémentaires.
- Rédige en français.

Réponds UNIQUEMENT avec un objet JSON valide (aucun texte autour), au format exact :
{
  "study": {"species": "", "body_part": "", "view": "", "image_quality": ""},
  "findings": ["observation 1", "observation 2"],
  "impression": "synthèse en une à trois phrases",
  "differentials": ["diagnostic différentiel le plus probable", "..."],
  "recommendations": ["examen ou conduite recommandée", "..."],
  "confidence": "faible | modérée | élevée",
  "limitations": "limites de cette interprétation"
}"""

DISCLAIMER = (
    "⚠️ **Brouillon généré par IA — à relire et valider par un vétérinaire qualifié. "
    "Ceci n'est PAS un diagnostic. À corréler avec les signes cliniques.**"
)


def get_radiology_prompt() -> str:
    return st.secrets.get("RADIOLOGY_PROMPT", DEFAULT_RADIOLOGY_PROMPT)


_CONTEXT_LABELS = {
    "species": "Espèce", "breed": "Race", "age": "Âge", "sex": "Sexe",
    "view": "Incidence", "body_part": "Région", "modality": "Modalité",
}


def _build_prompt(meta: dict, n_views: int = 1) -> str:
    parts = [get_radiology_prompt()]
    meta = meta or {}
    if n_views > 1:
        views = meta.get("views") or []
        labelled = ", ".join(
            f"Vue {i + 1}" + (f" ({v})" if v else "") for i, v in enumerate(views)
        ) or f"{n_views} vues"
        parts.append(
            f"Tu reçois {n_views} vues d'une même étude radiographique du même animal "
            f"({labelled}). Analyse-les ENSEMBLE : corrèle les projections, localise les "
            "lésions en 3D, et distingue les vraies lésions des artefacts de superposition. "
            "Produis UN SEUL compte rendu intégré ; référence les vues par leur projection "
            "quand c'est pertinent."
        )
    signalment = ", ".join(f"{lbl} : {meta[k]}" for k, lbl in _CONTEXT_LABELS.items() if meta.get(k))
    history = (meta.get("history") or "").strip()
    if signalment or history:
        block = [
            "Contexte clinique fourni par le vétérinaire (oriente l'analyse ; fonde tes "
            "observations sur l'image et signale toute incohérence avec ce contexte) :"
        ]
        if signalment:
            block.append(f"- Signalement : {signalment}")
        if history:
            block.append(f"- Motif / antécédents : {history}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def interpret_with_retry(images: list[bytes], model: str, meta: dict) -> dict:
    """Send one or more views of a study to Gemini and return the parsed report dict.

    Multiple images are interpreted together as a single integrated study (see
    _build_prompt). Mirrors streamlit_app.summarize_with_retry: rotate keys on
    quota errors, back off on transient 500s.
    """
    keys = common.get_gemini_keys()
    imgs = [Image.open(io.BytesIO(b)) for b in images]
    prompt = _build_prompt(meta, len(imgs))
    last_error: Exception | None = None

    for attempt in range(max(1, len(keys))):
        key = common.get_current_gemini_key()
        client = genai.Client(api_key=key)
        try:
            response = client.models.generate_content(
                model=model,
                contents=[*imgs, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return _parse_report(response.text)
        except Exception as e:
            last_error = e
            msg = str(e)
            transient = "500" in msg or "deadline" in msg.lower()
            quota_error = "RESOURCE_EXHAUSTED" in msg or "429" in msg or "UNAVAILABLE" in msg or "503" in msg
            if quota_error and attempt < len(keys) - 1:
                common.rotate_to_next_gemini_key()
                continue
            elif transient and attempt < max(1, len(keys)) - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    if last_error:
        raise last_error
    return {}


def _parse_report(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw[:4].lower() == "json":
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception:
            return {"_raw": text}


def build_report_md(report: dict) -> str:
    """Render the report dict to markdown, disclaimer banner first."""
    lines = [DISCLAIMER, ""]
    if "_raw" in report:
        lines += ["## Compte rendu (brut)", "", report.get("_raw", "")]
        return "\n".join(lines)

    study = report.get("study", {}) or {}
    sline = ", ".join(f"{k} : {v}" for k, v in study.items() if v)
    lines += ["## Technique / Étude", sline or "_Non précisé_", ""]

    lines += ["## Observations"]
    lines += [f"- {f}" for f in (report.get("findings") or [])] or ["- _Aucune décrite_"]

    lines += ["", "## Impression", report.get("impression") or "_Non précisée_", ""]

    lines += ["## Diagnostics différentiels"]
    lines += [f"{i + 1}. {d}" for i, d in enumerate(report.get("differentials") or [])] or ["- _Aucun_"]

    lines += ["", "## Recommandations"]
    lines += [f"- {r}" for r in (report.get("recommendations") or [])] or ["- _Aucune_"]

    lines += ["", f"**Niveau de confiance :** {report.get('confidence') or '_non précisé_'}", ""]
    lines += ["## Limites", report.get("limitations") or "_Non précisées_"]
    return "\n".join(lines)
