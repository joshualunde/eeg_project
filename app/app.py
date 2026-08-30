"""
EEG Mental Workload Classifier -- Streamlit demo app.

Loads the SVM model trained in Notebook 3 and classifies a 4-second,
14-channel EEG window as low / moderate / high mental workload.

Two ways to provide a sample:
1. Pick one of a few bundled example windows (guaranteed to work, no setup)
2. Upload your own .npy file shaped (14, 512) -- e.g. one exported from
   the project notebooks
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
from scipy.integrate import trapezoid
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Config / constants -- must match what was used in Notebooks 1-3 exactly,
# or the extracted features won't match what the model was trained on.
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "final_model.joblib")
METADATA_PATH = os.path.join(APP_DIR, "metadata.json")
SAMPLES_DIR = os.path.join(APP_DIR, "samples")

CLASS_NAMES = {0: "Low", 1: "Moderate", 2: "High"}
CLASS_COLORS = {0: "#4C9F70", 1: "#E8A33D", 2: "#D1495B"}


@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_data
def list_sample_files():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith(".npy"))


def compute_band_power(window, sfreq, bands):
    """Same logic as Notebook 2's compute_band_power -- must stay identical
    to how the model's training features were generated."""
    n_channels = window.shape[0]
    band_powers = {band: np.zeros(n_channels) for band in bands}
    for ch in range(n_channels):
        freqs, psd = welch(window[ch], fs=sfreq, nperseg=min(256, window.shape[1]))
        for band, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            band_powers[band][ch] = trapezoid(psd[mask], freqs[mask])
    return band_powers


def extract_features(window, sfreq, bands, channel_names):
    band_powers = compute_band_power(window, sfreq, bands)
    row = np.concatenate([band_powers[band] for band in bands])
    feature_names = [f"{band}_{ch}" for band in bands for ch in channel_names]
    return row.reshape(1, -1), feature_names


def plot_window(window, channel_names, sfreq):
    t = np.arange(window.shape[1]) / sfreq
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(8, 0.5 * len(channel_names)), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, window[i], linewidth=0.6, color="#3366CC")
        ax.set_ylabel(channel_names[i], rotation=0, ha="right", fontsize=6)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EEG Mental Workload Classifier", layout="wide")
st.title("EEG Mental Workload Classifier")
st.caption(
    "Classifies a 4-second, 14-channel EEG window as low, moderate, or high "
    "mental workload, using band-power features (theta/alpha/beta/gamma) "
    "grounded in cognitive load theory. Trained on the STEW dataset (n=45)."
)

model, metadata = load_model_and_metadata()
SFREQ = metadata["sfreq"]
CHANNEL_NAMES = metadata["channel_names"]
BANDS = {k: tuple(v) for k, v in metadata["bands"].items()}
FEATURE_NAMES = metadata["feature_names"]
EXPECTED_TIMEPOINTS = int(metadata["window_seconds"] * SFREQ)

st.sidebar.header("Provide an EEG sample")
input_mode = st.sidebar.radio("Input method", ["Pick an example", "Upload a .npy file"])

window = None
true_label_hint = None

if input_mode == "Pick an example":
    sample_files = list_sample_files()
    if not sample_files:
        st.sidebar.warning(
            "No example files found in the samples/ folder. "
            "Use the upload option, or add example .npy files to samples/."
        )
    else:
        chosen = st.sidebar.selectbox("Example window", sample_files)
        window = np.load(os.path.join(SAMPLES_DIR, chosen))
        # Example filenames are expected in the form "label{0,1,2}_subjXX_winYY.npy"
        # so the sidebar can show what the true label actually was, for context.
        if chosen.startswith("label"):
            true_label_hint = int(chosen[5])
else:
    uploaded = st.sidebar.file_uploader("Upload a (14, 512) EEG window as .npy", type=["npy"])
    if uploaded is not None:
        window = np.load(uploaded)

if window is None:
    st.info("Choose an example window or upload a file from the sidebar to get a prediction.")
    st.stop()

if window.shape != (len(CHANNEL_NAMES), EXPECTED_TIMEPOINTS):
    st.error(
        f"Expected shape ({len(CHANNEL_NAMES)}, {EXPECTED_TIMEPOINTS}) -- "
        f"got {window.shape}. This app expects a single 4-second, 14-channel "
        f"window at {SFREQ}Hz, matching the STEW preprocessing pipeline."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
features, feature_names_runtime = extract_features(window, SFREQ, BANDS, CHANNEL_NAMES)
pred_class = int(model.predict(features)[0])
pred_proba = model.predict_proba(features)[0]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    st.markdown(
        f"### <span style='color:{CLASS_COLORS[pred_class]}'>{CLASS_NAMES[pred_class]} workload</span>",
        unsafe_allow_html=True,
    )
    if true_label_hint is not None:
        st.caption(f"(This example's actual recorded label: {CLASS_NAMES[true_label_hint]})")

    st.write("Class probabilities:")
    for cls_idx, cls_name in CLASS_NAMES.items():
        st.progress(float(pred_proba[cls_idx]), text=f"{cls_name}: {pred_proba[cls_idx]:.2f}")

    st.caption(
        "Note: this is a research/portfolio demo trained on 45 subjects, not a "
        "validated clinical or occupational tool. Predictions should not be "
        "used to make real decisions about a person's mental state."
    )

with col2:
    st.subheader("Raw EEG window")
    fig = plot_window(window, CHANNEL_NAMES, SFREQ)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# SHAP explanation for this specific prediction
# ---------------------------------------------------------------------------
st.subheader("Why this prediction? (SHAP)")
st.caption(
    "Shows which band-power features pushed this specific prediction toward "
    "or away from the predicted class. Frontal theta/alpha features ranking "
    "highly here would echo the cognitive-load-theory pattern found during "
    "model development (see project README)."
)

try:
    # TreeExplainer only works for tree-based models (Random Forest). For the
    # deployed SVM, KernelExplainer is model-agnostic but much slower --
    # background sample kept tiny here to stay responsive in a live app.
    def extract_class_shap(shap_values, pred_class, sample_idx=0):
        """Handle both SHAP return formats: older versions return a list of
        per-class arrays shaped (n_samples, n_features); newer versions can
        return a single ndarray shaped (n_samples, n_features, n_classes).
        Indexing the wrong way causes an out-of-bounds error when n_samples=1."""
        if isinstance(shap_values, list):
            return shap_values[pred_class][sample_idx]
        else:
            return shap_values[sample_idx, :, pred_class]

    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        class_shap = extract_class_shap(shap_values, pred_class)
    else:
        background = np.zeros((1, features.shape[1]))  # zero background -- fast but coarse
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(features, nsamples=100)
        class_shap = extract_class_shap(shap_values, pred_class)

    order = np.argsort(np.abs(class_shap))[::-1][:10]
    fig2, ax = plt.subplots(figsize=(6, 4))
    colors = ["#D1495B" if v > 0 else "#4C9F70" for v in class_shap[order]]
    ax.barh([feature_names_runtime[i] for i in order][::-1], class_shap[order][::-1], color=colors[::-1])
    ax.set_xlabel(f"SHAP value (impact on '{CLASS_NAMES[pred_class]}' prediction)")
    fig2.tight_layout()
    st.pyplot(fig2)
except Exception as e:
    st.warning(f"SHAP explanation unavailable for this run ({e}).")

st.divider()
st.caption(
    "Built as part of an end-to-end EEG mental workload classification project. "
    "See the project repository for the full preprocessing, modeling, and "
    "evaluation pipeline, including subject-independent cross-validation and "
    "a documented data-leakage correction."
)
