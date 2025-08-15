# MGCI Composite Scores — Streamlit App

An interactive Streamlit app to compute and visualize **Model-Guided Clinical Indices (MGCI)** from speech/linguistic biomarkers.  
MGCI are SHAP-informed, z-scored, **per-category composite scores** with options for top-_k_ feature selection, L1 normalization, and signed sub-composites (AD-like / CN-like). The app provides feature-level KDE plots with group overlays and individual percentile readouts, plus CSV export.

---

## ✨ What the app does

- **Top-k distinctive features (per category):** rank by mean |SHAP| and pick the most informative biomarkers.
- **Composites:** compute category scores using SHAP-weighted z-scores; optionally split into +/− sub-composites.
- **Distributions:** smoothed KDE plots by group (e.g., CN vs AD) with a **vertical line and percentiles** for a selected individual.
- **Feature viewer:** density plots for any single feature with adjustable smoothing.
- **Export:** download per-participant composite scores as CSV (single composite or split +/−, with custom labels).

---

## 🗂️ Repository layout

