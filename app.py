"""
Week 2, Days 9-10 — Streamlit skeleton.

Run locally (NOT executed here):
    streamlit run app.py

Scope for this week: disease selection, structured input (form or CSV),
run inference, display the raw prediction + confidence + global feature
importance. NOT in scope yet: the plain-language AI report (Week... later),
the Ollama chatbot, and RAG (Week 3).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import sys
sys.path.append(str(Path(__file__).parent / "src"))
from chatbot import stream_patient_answer

MODELS_DIR = Path(__file__).parent / "models"

# --- Domain-knowledge value sets for the categorical fields in each dataset ---
CKD_CATEGORICAL_OPTIONS = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["present", "notpresent"],
    "ba": ["present", "notpresent"],
    "htn": ["yes", "no"],
    "dm": ["yes", "no"],
    "cad": ["yes", "no"],
    "appet": ["good", "poor"],
    "pe": ["yes", "no"],
    "ane": ["yes", "no"],
}
LIVER_CATEGORICAL_OPTIONS = {
    "Gender": ["Male", "Female"],
}

# --- Reasonable starting defaults / step sizes / tooltips, purely for a usable demo ---
NUMERIC_FIELD_CONFIG = {
    # CKD
    "age": {"default": 45.0, "step": 1.0, "help": "Age in years"},
    "bp": {"default": 80.0, "step": 1.0, "help": "Blood pressure (mm/Hg)"},
    "sg": {"default": 1.02, "step": 0.005, "help": "Urine specific gravity"},
    "al": {"default": 0.0, "step": 1.0, "help": "Albumin level in urine (0-5 scale)"},
    "su": {"default": 0.0, "step": 1.0, "help": "Sugar level in urine (0-5 scale)"},
    "bgr": {"default": 100.0, "step": 1.0, "help": "Random blood glucose (mg/dL)"},
    "bu": {"default": 40.0, "step": 1.0, "help": "Blood urea (mg/dL)"},
    "sc": {"default": 1.0, "step": 0.1, "help": "Serum creatinine (mg/dL)"},
    "sod": {"default": 140.0, "step": 1.0, "help": "Sodium (mEq/L)"},
    "pot": {"default": 4.5, "step": 0.1, "help": "Potassium (mEq/L)"},
    "hemo": {"default": 14.0, "step": 0.1, "help": "Hemoglobin (g/dL)"},
    "pcv": {"default": 40.0, "step": 1.0, "help": "Packed cell volume (%)"},
    "wc": {"default": 8000.0, "step": 100.0, "help": "White blood cell count (cells/cumm)"},
    "rc": {"default": 5.0, "step": 0.1, "help": "Red blood cell count (millions/cmm)"},
    # Liver
    "Age": {"default": 45.0, "step": 1.0, "help": "Age in years"},
    "Total_Bilirubin": {"default": 1.0, "step": 0.1, "help": "Total bilirubin (mg/dL)"},
    "Direct_Bilirubin": {"default": 0.3, "step": 0.1, "help": "Direct bilirubin (mg/dL)"},
    "Alkaline_Phosphotase": {"default": 200.0, "step": 1.0, "help": "Alkaline phosphatase (IU/L)"},
    "Alamine_Aminotransferase": {"default": 30.0, "step": 1.0, "help": "ALT / SGPT (IU/L)"},
    "Aspartate_Aminotransferase": {"default": 30.0, "step": 1.0, "help": "AST / SGOT (IU/L)"},
    "Total_Protiens": {"default": 6.5, "step": 0.1, "help": "Total proteins (g/dL)"},
    "Albumin": {"default": 3.5, "step": 0.1, "help": "Albumin (g/dL)"},
    "Albumin_and_Globulin_Ratio": {"default": 1.0, "step": 0.1, "help": "Albumin/Globulin ratio"},
}

DISEASES = {
    "Chronic Kidney Disease": {
        "key": "kidney",
        "pipeline_file": "kidney_pipeline.joblib",
        "metadata_file": "kidney_pipeline_metadata.json",
        "importance_file": "kidney_feature_importance.json",
        "categorical_options": CKD_CATEGORICAL_OPTIONS,
        "positive_label": "CKD (Chronic Kidney Disease) indicated",
        "negative_label": "No CKD indicated",
    },
    "Liver Disease": {
        "key": "liver",
        "pipeline_file": "liver_pipeline.joblib",
        "metadata_file": "liver_pipeline_metadata.json",
        "importance_file": "liver_feature_importance.json",
        "categorical_options": LIVER_CATEGORICAL_OPTIONS,
        "positive_label": "Liver disease indicated",
        "negative_label": "No liver disease indicated",
    },
}


@st.cache_resource
def load_artifacts(pipeline_file, metadata_file, importance_file):
    pipeline_path = MODELS_DIR / pipeline_file
    metadata_path = MODELS_DIR / metadata_file
    importance_path = MODELS_DIR / importance_file

    if not pipeline_path.exists():
        return None, {}, []

    pipeline = joblib.load(pipeline_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    importance = json.loads(importance_path.read_text()) if importance_path.exists() else []
    return pipeline, metadata, importance


def pretty(col):
    return col.replace("_", " ").strip().title()


def render_manual_form(disease_cfg, metadata):
    numeric_cols = metadata.get("numeric_features", [])
    categorical_cols = metadata.get("categorical_features", [])
    options = disease_cfg["categorical_options"]
    key_prefix = disease_cfg["key"]

    values = {}

    st.markdown("**Numeric values**")
    cols_per_row = 2
    for i in range(0, len(numeric_cols), cols_per_row):
        row_cols = numeric_cols[i:i + cols_per_row]
        st_cols = st.columns(len(row_cols))
        for st_col, colname in zip(st_cols, row_cols):
            cfg = NUMERIC_FIELD_CONFIG.get(colname, {"default": 0.0, "step": 0.1, "help": None})
            with st_col:
                values[colname] = st.number_input(
                    pretty(colname),
                    min_value=0.0,
                    value=cfg["default"],
                    step=cfg["step"],
                    help=cfg.get("help"),
                    key=f"{key_prefix}_{colname}",
                )

    if categorical_cols:
        st.markdown("**Categorical values**")
        for i in range(0, len(categorical_cols), cols_per_row):
            row_cols = categorical_cols[i:i + cols_per_row]
            st_cols = st.columns(len(row_cols))
            for st_col, colname in zip(st_cols, row_cols):
                with st_col:
                    opts = options.get(colname, ["unknown"])
                    values[colname] = st.selectbox(
                        pretty(colname), opts, key=f"{key_prefix}_{colname}"
                    )

    return values


def run_prediction(pipeline, X):
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None
    return pred, proba


def show_single_result(disease_cfg, pred, proba):
    """Styled result card: color-coded badge, icon, and a confidence bar."""
    is_positive = pred == 1
    label = disease_cfg["positive_label"] if is_positive else disease_cfg["negative_label"]
    icon = "⚠️" if is_positive else "✅"
    badge_color = "#B3261E" if is_positive else "#1F5C4E"
    badge_bg = "#FBE9E7" if is_positive else "#E8F1EE"

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:28px;">{icon}</span>
                <span style="
                    background-color:{badge_bg};
                    color:{badge_color};
                    font-weight:600;
                    padding:4px 12px;
                    border-radius:999px;
                    font-size:15px;">
                    {label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if proba is not None:
            st.caption("Model confidence (probability of disease)")
            st.progress(min(max(proba, 0.0), 1.0), text=f"{proba:.1%}")


def render_sidebar_summary():
    """Always-visible sidebar card showing the most recent result per disease,
    across reruns and even after switching between diseases."""
    summaries = []
    for name, cfg in DISEASES.items():
        ctx = st.session_state.get(f"{cfg['key']}_context")
        if ctx:
            summaries.append((name, ctx))

    if not summaries:
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🗂️ Last results this session**")
    for name, ctx in summaries:
        with st.sidebar.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(ctx["prediction_label"])
            if ctx.get("confidence") is not None:
                st.caption(f"Confidence: {ctx['confidence']:.1%}")


def main():
    st.set_page_config(page_title="Healthcare AI Monitoring App", page_icon="🩺", layout="centered")
    st.title(":Stethos: Healthcare AI Monitoring System")
    st.caption("Caution: This application is not a medical diagnosis. \
        This application is only meant as a estimation.\
        Approach only qualified medical professional for actual diagnosis."
    )

    disease_name = st.sidebar.selectbox("Select disease / report type", list(DISEASES.keys()))
    disease_cfg = DISEASES[disease_name]

    pipeline, metadata, importance = load_artifacts(
        disease_cfg["pipeline_file"], disease_cfg["metadata_file"], disease_cfg["importance_file"]
    )

    if pipeline is None:
        st.error(
            f"No trained model found for **{disease_name}**. "
            f"Expected `models/{disease_cfg['pipeline_file']}` — run the training "
            f"(and optionally tuning) script for this disease first."
        )
        return

    tuned_f1 = metadata.get("tuning", {}).get("tuned_test_f1")
    baseline_f1 = metadata.get("test_f1")
    f1_display = f"{tuned_f1:.4f} (tuned)" if tuned_f1 is not None else f"{baseline_f1:.4f} (baseline)"
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Model:** {metadata.get('chosen_model', 'unknown')}")
    st.sidebar.caption(f"**Test F1:** {f1_display}")

    render_sidebar_summary()

    input_mode = st.radio("Input method", ["Fill form", "Upload CSV"], horizontal=True)

    if input_mode == "Fill form":
        values = render_manual_form(disease_cfg, metadata)
        if st.button("Run prediction", type="primary"):
            X = pd.DataFrame([values])
            pred, proba = run_prediction(pipeline, X)
            confidence = float(proba[0]) if proba is not None else None
            show_single_result(disease_cfg, pred[0], confidence)

            # Store this result as the active patient context for the chatbot below
            st.session_state[f"{disease_cfg['key']}_context"] = {
                "disease": disease_name,
                "values": values,
                "prediction_label": disease_cfg["positive_label"] if pred[0] == 1 else disease_cfg["negative_label"],
                "confidence": confidence,
            }

    else:
        expected_cols = metadata.get("numeric_features", []) + metadata.get("categorical_features", [])
        st.caption(f"CSV must contain columns: {', '.join(expected_cols)}")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing expected columns: {missing}")
            else:
                pred, proba = run_prediction(pipeline, df[expected_cols])
                result_df = df.copy()
                result_df["prediction_label"] = [
                    disease_cfg["positive_label"] if p == 1 else disease_cfg["negative_label"]
                    for p in pred
                ]
                if proba is not None:
                    result_df["confidence"] = proba
                st.dataframe(result_df)
                st.download_button(
                    "Download results as CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{disease_cfg['key']}_predictions.csv",
                )

    if importance:
        with st.expander("Top contributing features (global model importance)"):
            imp_df = pd.DataFrame(importance[:10]).set_index("feature")
            st.bar_chart(imp_df)

    st.markdown("---")
    st.subheader(":speech_balloon: Ask about your results")

    context_key = f"{disease_cfg['key']}_context"
    history_key = f"{disease_cfg['key']}_chat_history"
    patient_context = st.session_state.get(context_key)

    if patient_context is None:
        st.info(
            "Run a prediction above first (Fill form mode) so the assistant "
            "has your report to talk about. You can still ask general "
            "questions about lab markers without one."
        )

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for turn in st.session_state[history_key]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input(f"Ask a question about your {disease_name.lower()} report...")
    if question:
        st.session_state[history_key].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                stream = stream_patient_answer(
                    question=question,
                    disease_key=disease_cfg["key"],
                    patient_context=patient_context,
                    chat_history=st.session_state[history_key][:-1],
                )
                reply = st.write_stream(stream)
            except Exception as e:
                reply = (
                    "Sorry, I couldn't reach the chatbot. Make sure Ollama is "
                    f"running locally (`ollama serve`) and the vector store has "
                    f"been built (`python src/build_vector_store.py`).\n\nDetails: {e}"
                )
                st.markdown(reply)
        st.session_state[history_key].append({"role": "assistant", "content": reply})

    st.markdown("---")
    st.caption(
        ":warning: This tool provides a decision-support estimate only and is not a "
        "medical diagnosis. Consult a qualified healthcare professional for "
        "interpretation of your results."
    )


if __name__ == "__main__":
    main()
