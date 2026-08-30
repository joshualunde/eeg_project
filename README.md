# EEG Mental Workload Classifier -- Deployment App

## What's here

- `app.py` -- the Streamlit app
- `requirements.txt` -- Python dependencies
- `final_model.joblib` -- **you need to copy this in** from Notebook 3's output (`{PROJECT_DIR}/models/final_model.joblib`)
- `metadata.json` -- **you need to copy this in** from Notebook 3's output (`{PROJECT_DIR}/models/metadata.json`)
- `samples/` -- a few example EEG windows for the "Pick an example" dropdown. **You need to generate these** (see below).

## Step 1: Get your model files into this folder

From your Google Drive (`eeg-workload-project/models/`), download:
- `final_model.joblib`
- `metadata.json`

and place both directly in this `app/` folder (same level as `app.py`).

## Step 2: Generate a few example windows

Run this in a new cell in Notebook 2 or 3 (anywhere `X_windows`, `y_windows`, and `subj_windows` are loaded), then download the resulting files from Drive into this project's `app/samples/` folder:

```python
import os
SAMPLES_EXPORT_DIR = f'{PROJECT_DIR}/samples_for_app'
os.makedirs(SAMPLES_EXPORT_DIR, exist_ok=True)

# Pick a few windows spanning each class for a varied demo
rng = np.random.default_rng(42)
for label in [0, 1, 2]:
    label_indices = np.where(y_windows == label)[0]
    chosen = rng.choice(label_indices, size=2, replace=False)
    for j, idx in enumerate(chosen):
        fname = f'label{label}_subj{subj_windows[idx]:02d}_win{j}.npy'
        np.save(os.path.join(SAMPLES_EXPORT_DIR, fname), X_windows[idx])

print('Saved example windows to', SAMPLES_EXPORT_DIR)
print(os.listdir(SAMPLES_EXPORT_DIR))
```

This saves 6 example windows (2 per class), with the filename format `label{N}_subjXX_winY.npy` -- the app reads the leading `label{N}` to show you the true recorded label alongside the prediction, so you can see at a glance whether the model got it right.

Download the resulting files from Drive and place them in this project's `app/samples/` folder.

## Step 3: Test locally (optional but recommended)

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Confirm both input modes work before deploying.

## Step 4: Deploy to Streamlit Community Cloud

1. Push this `app/` folder (with `final_model.joblib`, `metadata.json`, and populated `samples/` included) to a GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", select your repository, set the main file path to `app.py`.
4. Deploy. Streamlit will install from `requirements.txt` automatically.

Your app will be live at a public `*.streamlit.app` URL you can put directly on your resume/portfolio.

## Known limitations to mention in your write-up

- The app skips ICA-based artifact removal at inference time -- ICA needs a full recording to fit reliably, which isn't available for a single short window a user might upload. The bundled example windows come pre-cleaned from the full pipeline; a freshly uploaded raw window will not have artifacts removed. Worth stating this explicitly rather than presenting the live upload path as fully equivalent to the notebook pipeline.
- SHAP explanations for the SVM use `KernelExplainer` with a zero-vector background for speed in a live app -- this is coarser than the `TreeExplainer` analysis used in Notebook 3's proper evaluation. Treat the in-app explanation as illustrative, and point to the notebook's more rigorous SHAP analysis as the real result.
