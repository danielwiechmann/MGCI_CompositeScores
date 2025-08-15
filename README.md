# MGCI Composite Scores — Streamlit App

This repository contains a Streamlit app to compute and visualize **Model-Guided Clinical Indices (MGCI)** from speech/linguistic biomarkers.  
MGCI are SHAP-informed, z-scored **per-category composite scores** with options for top-k feature selection, L1 normalization, signed sub-composites (+/−), KDE plots, percentile readouts, and CSV export.

---

## Features

- **Top-k features per category:** rank by mean |SHAP| and choose how many to include.
- **Composite scores:** SHAP-weighted z-scores; optional split into “AD-like (+)” and “CN-like (−)”.
- **Distributions:** group KDE plots with an individual’s vertical line and CN/AD percentiles.
- **Single-feature viewer:** adjustable smoothing bandwidth.
- **Export:** per-participant composites to CSV (single or split +/− with custom labels).

---

## Repository layout

- app4_web.py # Streamlit app (entry point)
- requirements.txt # Python dependencies
- LICENSE

> Users supply their own CYMO-generated data (see “Data expected”).

---

## Run locally

```bash
# optional but recommended
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app4_web.py
