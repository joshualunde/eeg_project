# EEG-Based Mental Workload Classifier

An end-to-end machine learning pipeline that classifies mental workload (low / moderate / high) from raw EEG signal, built to explore how cognitive load theory can inform signal processing and feature engineering for a genuinely deployed ML system.

**Live demo:** https://eegproject-yhe24iij4vtukex65gwkjq.streamlit.app/
**Stack:** Python, MNE, scikit-learn, PyTorch, SHAP, Streamlit

---

## Why this project

Most portfolio ML projects show either a modeling notebook or a deployed app, rarely both with real signal-processing depth underneath. This project was built to combine three things: genuine EEG signal processing (filtering, artifact removal), domain-grounded feature engineering informed by cognitive load theory, and honest, rigorous evaluation — including a documented methodological mistake and its fix, which is arguably more informative than a clean success story would be.

## Dataset

[STEW (Simultaneous Task EEG Workload)](https://www.kaggle.com/datasets/mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset) — 45 subjects, 14-channel EEG (Emotiv EPOC, 128Hz), each recorded during a 150-second multitasking test (SIMKAP), with a post-task self-reported workload rating (1-9) mapped to a 3-class label (low / moderate / high).

**A key structural fact that shaped several decisions below:** each subject contributes exactly one recording and one label — there is no within-subject low-vs-high contrast available, only a between-subject comparison.

## Pipeline

1. **Bandpass filtering** (4-45Hz) to remove drift and high-frequency noise, isolating the theta/alpha/beta/gamma range associated with cognitive state.
2. **ICA-based artifact removal.** STEW has no dedicated EOG (eye-movement) channel, so a frontal channel (AF3) was used as a pseudo-EOG proxy. Component selection went through three iterations before landing on a stable method — see [Methodological Notes](#methodological-notes) below.
3. **Windowing** into 4-second, 50%-overlap segments (74 windows per subject, 3,330 total), turning 45 long recordings into a workable number of training examples while preserving subject identity for evaluation.
4. **Band-power feature extraction** — theta, alpha, beta, and gamma power per channel (56 features per window), directly motivated by cognitive load theory's predictions about frontal theta and parietal/occipital alpha.
5. **Modeling** — Random Forest and SVM baselines on engineered features, plus a 1D-CNN trained directly on raw filtered windows for comparison.
6. **Evaluation** — subject-independent `GroupKFold` cross-validation throughout, so no model is ever evaluated on a subject it saw during training.
7. **Explainability** — SHAP applied to the trained models to check whether learned feature importance actually aligns with cognitive load theory's predictions.
8. **Deployment** — the final SVM model served via a Streamlit app, with example EEG windows and live SHAP explanations per prediction.

## Results

| Model | Subject-independent accuracy | Naive (leaky) accuracy | Leakage inflation |
|---|---|---|---|
| Random Forest | 0.375 | 0.958 | +0.583 |
| SVM | 0.470 | 0.493 | +0.024 |
| 1D-CNN (raw signal) | 0.306 | — | — |

Chance level (3-class): 0.333. **SVM was selected for deployment**, both for its higher subject-independent accuracy and its far smaller leakage inflation — a signal that it generalizes on genuine workload patterns rather than partially memorizing subject-specific quirks.

Binary framing (low vs. high, dropping the moderate class) was also tested and rejected: it cleared its 0.500 chance level by only +0.013 with high variance, a *weaker* relative result than the 3-class model's +0.042 margin over its own chance level — despite binary being the more common published framing for this dataset.

## Key findings

### 1. Data leakage can inflate EEG classification accuracy dramatically
Comparing subject-independent cross-validation against a naive random split (where windows from the same subject can appear in both train and test) showed Random Forest accuracy inflate from 0.375 to 0.958 — a textbook demonstration of a common, easy-to-miss pitfall in EEG ML research. SVM showed almost no inflation (+0.024), suggesting its smoother decision boundary is less prone to memorizing subject-specific signal than Random Forest's more flexible tree structure.

### 2. Per-subject feature normalization is the wrong tool for this dataset's design
A natural instinct when comparing EEG power across subjects is to normalize each subject's features against their own baseline, since absolute EEG power varies enormously due to skull thickness, electrode contact, and other anatomical factors unrelated to cognitive state. This was tried and explicitly reverted: because each subject in STEW has only *one* label (no within-subject contrast to preserve), per-subject z-scoring removes exactly the between-subject variance that carries all of the label-relevant information, rather than isolating a real effect. This is a design-level incompatibility, not a bug — it would be the right technique for a paired within-subject study, but not for this one.

### 3. Domain-informed features outperformed a learned representation at this sample size
A 1D-CNN trained directly on raw filtered signal underperformed the hand-engineered band-power features (0.306 vs. 0.470 subject-independent accuracy), consistent with the well-established pattern that deep learning needs substantially more data than classical ML with strong domain-informed features to show an advantage. At n=45, the theory-driven features won.

### 4. SHAP explainability independently supports (with a caveat) the cognitive-load-theory framing
SHAP analysis of the trained model's predictions showed frontal alpha and theta power (`alpha_AF3`, `theta_AF3`) among the most influential features — consistent with cognitive load theory's predictions about frontal midline activity during high mental workload. **Caveat:** AF3 was also the channel used as the pseudo-EOG reference for artifact removal, so some of its outsized importance may reflect residual eye-movement artifact rather than purely cognitive signal. This is worth stating plainly rather than presenting the SHAP result as unambiguous confirmation.

## Methodological notes

The ICA-based artifact removal step went through three iterations, which is itself a useful record of a common EEG preprocessing pitfall:

- **v1 (topography-based):** flagged components by frontal-vs-posterior spatial weighting. Verified visually to remove nothing effective — topography alone isn't a reliable enough signal, since genuine frontal brain activity (like theta) also loads heavily on frontal channels.
- **v2 (correlation threshold):** used `find_bads_eog` with a z-score threshold against a pseudo-EOG channel. Threshold=3.0 was too conservative (no cleaning effect on ~half of subjects); threshold=1.5 overcorrected, removing real neural signal and dropping subject-independent accuracy *below chance*.
- **v3 (rank-based, final):** removes exactly the single component most correlated with the pseudo-EOG channel per subject, avoiding the instability of an arbitrary threshold. Correlations were consistently high (mean 0.80 across all 45 subjects), and subject-independent model accuracy recovered to a stable, above-chance range.

## Limitations

- **Small sample size (n=45).** Results should be read as a proof-of-concept, not a clinically or occupationally validated tool. Fold-to-fold variance in cross-validation was often substantial.
- **No true EOG channel.** Artifact removal relies on a single frontal EEG channel as an imperfect proxy, which likely leaves some residual contamination and complicates cleanly separating genuine frontal theta/alpha signal from eye-movement artifact (see SHAP caveat above).
- **One label per subject.** This constrains what evaluation and normalization strategies are valid, as detailed above, and limits the model to learning purely between-subject patterns.
- **The deployed app skips ICA at inference time** for freshly uploaded windows, since ICA requires a full recording to fit reliably and isn't practical to run on a single short window submitted live. Bundled example windows are pre-cleaned from the full pipeline; user-uploaded windows are not.

## Repository structure

```
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_and_baseline_model.ipynb
│   └── 03_deep_learning_and_explainability.ipynb
├── app/
│   ├── app.py
│   ├── requirements.txt
│   ├── final_model.joblib
│   ├── metadata.json
│   └── samples/
└── README.md
```

## Running locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Acknowledgments

STEW dataset: Lim, W.L., Sourina, O., Wang, L.P. (2018). *STEW: Simultaneous Task EEG Workload Dataset.* IEEE DataPort.
