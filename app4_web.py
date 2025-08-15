# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Prefer SciPy; fall back to lightweight implementations if unavailable.
try:
    from scipy.stats import zscore, gaussian_kde  # pip install scipy
except Exception:
    # --- Fallbacks if SciPy isn't available ---
    def zscore(x, ddof=0):
        """Column-wise z-score for pandas objects or numpy arrays (population SD by default)."""
        import pandas as pd
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return (x - x.mean()) / x.std(ddof=ddof)
        x = np.asarray(x, dtype=float)
        return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0, ddof=ddof)

    class gaussian_kde:
        """Simple 1D Gaussian KDE fallback with Scott’s rule."""
        def __init__(self, data):
            d = np.asarray(data, dtype=float)
            d = d[np.isfinite(d)]
            if d.size < 2:
                d = np.array([0.0, 1.0])  # avoid degeneracy
            self.data = d
            self.n = d.size
            self.factor = self.n ** (-1.0 / 5.0)  # Scott’s factor
            sd = np.nanstd(d)
            self._bw = (sd * self.factor) if sd > 0 else 1.0

        def set_bandwidth(self, bw):
            try:
                bw = float(bw)
                if np.isfinite(bw) and bw > 0:
                    self._bw = bw
            except Exception:
                pass

        def __call__(self, x):
            x = np.asarray(x, dtype=float)
            bw = self._bw
            denom = bw * np.sqrt(2 * np.pi)
            diffs = (x[:, None] - self.data[None, :]) / bw
            return np.nanmean(np.exp(-0.5 * diffs**2) / denom, axis=1)


# -----------------------------
# Utilities for composites
# -----------------------------
def _select_topk_by_category(df_sorted: pd.DataFrame, feature_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return df_meta filtered to features present in data and top-k per category by Mean |SHAP value|."""
    valid = df_sorted[df_sorted["Feature"].isin(feature_df.columns)].copy()
    valid["abs_shap"] = (
        valid["Mean |SHAP value|"] if "Mean |SHAP value|" in valid.columns else valid["Mean SHAP value"].abs()
    )
    parts = []
    for c in sorted(valid["category"].dropna().unique()):
        d = valid[valid["category"] == c].sort_values("abs_shap", ascending=False).head(int(k))
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else valid.iloc[0:0].copy()


def _available_l1_denom(Z_sub: pd.DataFrame, weights_abs: np.ndarray) -> pd.Series:
    """Sum of |weights| for features that are present (non-NaN) for each row."""
    avail = Z_sub.notna().astype(float)
    den = avail.dot(np.abs(weights_abs))
    return den.replace(0.0, np.nan)


def compute_all_composites(
    df_sorted: pd.DataFrame,
    feature_df: pd.DataFrame,
    id_col: str = "TID",
    top_k: int = 5,
    normalize_l1: bool = True,
):
    """
    Compute per-category composites:
      - Magnitude (deviation): sum |s_f|*z  [optionally / sum|s_f|]
      - Directional signed (two subcategories):
          ADlike = sum_{s>0} s_f*z
          CNlike = - sum_{s<0} s_f*z
    Returns dict of DataFrames + meta + z-scores.
    """
    # 1) Top-k meta per category
    meta = _select_topk_by_category(df_sorted, feature_df, top_k)
    if meta.empty:
        raise ValueError("No overlapping features after top-k selection.")

    # 2) Z-score all valid features (population SD, ddof=0)
    feats = meta["Feature"].tolist()
    Z = feature_df[feats].apply(zscore, ddof=0)

    # 3) SHAP series
    s = meta.set_index("Feature")["Mean SHAP value"]

    cats = sorted(meta["category"].dropna().unique())

    # Prepare outputs
    mag = {}
    dir_AD = {}
    dir_CN = {}

    for c in cats:
        feats_c = meta.loc[meta["category"] == c, "Feature"].tolist()
        if not feats_c:
            for d in (mag, dir_AD, dir_CN):
                d[c] = pd.Series(index=feature_df.index, dtype=float)
            continue

        Zc = Z[feats_c]
        s_c = s.reindex(feats_c).values
        s_abs_c = np.abs(s_c)

        # --- Magnitude (deviation) ---
        num_mag = (Zc * s_abs_c).sum(axis=1)
        if normalize_l1:
            den_mag = _available_l1_denom(Zc, s_abs_c)
            mag[c] = num_mag / den_mag
        else:
            mag[c] = num_mag

        # --- Directional signed ---
        pos_mask = s_c > 0
        neg_mask = s_c < 0

        # ADlike
        if pos_mask.any():
            Z_pos = Zc.loc[:, np.array(feats_c)[pos_mask]]
            w_pos = s_c[pos_mask]
            num_ad = (Z_pos * w_pos).sum(axis=1)
            if normalize_l1:
                den_ad = _available_l1_denom(Z_pos, np.abs(w_pos))
                dir_AD[c] = num_ad / den_ad
            else:
                dir_AD[c] = num_ad
        else:
            dir_AD[c] = pd.Series(index=feature_df.index, dtype=float)

        # CNlike (positive magnitude)
        if neg_mask.any():
            Z_neg = Zc.loc[:, np.array(feats_c)[neg_mask]]
            w_neg = s_c[neg_mask]  # negative
            num_cn = - (Z_neg * w_neg).sum(axis=1)  # flip sign to be positive CN-likeness
            if normalize_l1:
                den_cn = _available_l1_denom(Z_neg, np.abs(w_neg))
                dir_CN[c] = num_cn / den_cn
            else:
                dir_CN[c] = num_cn
        else:
            dir_CN[c] = pd.Series(index=feature_df.index, dtype=float)

    # Build DataFrames
    def _to_df(dct):
        df = pd.DataFrame({c: dct[c] for c in cats})
        df[id_col] = feature_df[id_col].values
        df["group"] = feature_df["group"].values
        return df

    outputs = {
        "magnitude": _to_df(mag),
        "dir_ADlike": _to_df(dir_AD),
        "dir_CNlike": _to_df(dir_CN),
        "meta_topk": meta,   # keep for details tables
        "Z_all": Z           # z-scores for per-feature expansions
    }
    return outputs


# === Streamlit page ===
st.set_page_config(layout="wide", page_title="Model-Guided Clinical Indices (MGCI)", page_icon="🧠")
st.title("Model-Guided Clinical Indices (MGCI)")

# ---- File selection (replaces hardcoded paths) ----
st.sidebar.header("Upload required data")
df_sorted_file = st.sidebar.file_uploader(
    "df_sorted.csv (feature metadata + SHAP + category)",
    type=["csv"],
    help="Must include columns: Feature, Mean SHAP value, Mean |SHAP value| (or will be derived), category."
)
merged_df_file = st.sidebar.file_uploader(
    "merged_df.csv (per-participant measurements)",
    type=["csv"],
    help="Must include columns: TID, group, and your feature columns."
)
st.sidebar.markdown("---")
st.sidebar.caption("Optional: transcripts (used only in the viewer)")
text_train_file = st.sidebar.file_uploader("ADReSSo2021_train_CYMOinput.csv", type=["csv"])
text_test_file  = st.sidebar.file_uploader("ADReSSo2021_test_CYMOinput.csv", type=["csv"])

if (df_sorted_file is None) or (merged_df_file is None):
    st.info("Please upload **df_sorted.csv** and **merged_df.csv** in the sidebar to continue.")
    st.stop()

with st.spinner("Loading data..."):
    df_sorted = pd.read_csv(df_sorted_file)
    merged_df = pd.read_csv(merged_df_file)
    merged_df = merged_df[[c for c in merged_df.columns if not str(c).startswith("TOP")]]

    # Ensure required columns exist
    if "Mean |SHAP value|" not in df_sorted.columns and "Mean SHAP value" in df_sorted.columns:
        df_sorted["Mean |SHAP value|"] = df_sorted["Mean SHAP value"].abs()

    features = df_sorted["Feature"].values

    # Optional transcripts
    text_frames = []
    if text_train_file is not None:
        try:
            tt = pd.read_csv(text_train_file)[["TID", "text"]]
            text_frames.append(tt)
        except Exception:
            pass
    if text_test_file is not None:
        try:
            tt = pd.read_csv(text_test_file)[["TID", "text"]]
            text_frames.append(tt)
        except Exception:
            pass
    text_df = pd.concat(text_frames, ignore_index=True) if text_frames else pd.DataFrame(columns=["TID", "text"])

# -----------------------------
# Styling helpers (scientific look)
# -----------------------------
PLOT_FONT = dict(family="Helvetica, Arial, sans-serif", size=12)
GRID_COLOR = "rgba(0,0,0,0.12)"
AXIS_COLOR = "rgba(0,0,0,0.7)"

# Consistent colors per group (line + semi-transparent fill)
GROUP_COLORS = {
    "cn":  ("rgba(0,123,255,1.0)", "rgba(0,123,255,0.18)"),   # blue
    "ad":  ("rgba(220,53,69,1.0)",  "rgba(220,53,69,0.18)"),  # red
    "mci": ("rgba(255,193,7,1.0)",  "rgba(255,193,7,0.18)"),  # amber
}
DEFAULT_LINE = "rgba(108,117,125,1.0)"   # gray
DEFAULT_FILL = "rgba(108,117,125,0.18)"

def color_for_group(g):
    key = str(g).lower()
    return GROUP_COLORS.get(key, (DEFAULT_LINE, DEFAULT_FILL))

def percentile_rank(arr, v):
    """Percentile of v relative to arr using <= definition; returns np.nan if insufficient data."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return 100.0 * (np.sum(arr <= v) / arr.size)

# ============================================================
# TOP-K DISTINCTIVE FEATURES (at top)
# ============================================================
st.subheader("Top-k Distinctive Features by Category")

cat_options = sorted(df_sorted["category"].dropna().unique())
col_topk1, col_topk2, col_topk3 = st.columns([1.4, 1.0, 1.2])

with col_topk1:
    cat_selected = st.selectbox(
        "1) Choose a category",
        cat_options,
        help="The entire Composite Score section below will focus **only** on this category."
    )

with col_topk2:
    max_k_for_cat = int((df_sorted["category"] == cat_selected).sum())
    k_selected = st.slider(
        "2) How many top features (k) should define the category composite?",
        1, max_k_for_cat, min(5, max_k_for_cat), 1,
        help="Features are ranked **within this category** by Mean |SHAP|. The top-k features are also used to compute the composite(s)."
    )

with col_topk3:
    norm_l1 = st.checkbox(
        "3) Normalize by total |weight| (L1)",
        value=True,
        help="If checked, each composite is divided by the sum of |SHAP weights| of the features present for that person. "
             "This improves comparability when some features are missing."
    )

# Show top-k table for the selected category
topk_table = (
    df_sorted[df_sorted["category"] == cat_selected]
    .loc[lambda d: d["Feature"].isin(merged_df.columns)]
    .sort_values("Mean |SHAP value|", ascending=False)
    .head(k_selected)[["Feature", "Mean SHAP value", "Mean |SHAP value|"]]
)
st.dataframe(topk_table, use_container_width=True)

# Compute composites (all categories), then DISPLAY ONLY the selected category
all_scores = compute_all_composites(df_sorted, merged_df, top_k=k_selected, normalize_l1=norm_l1)

# DEBUG group means for the selected category only
st.caption("Group means (selected category; magnitude/deviation composite):")
score_cols = [c for c in all_scores["magnitude"].select_dtypes(include="number").columns if c != "TID"]
group_means_sel = (
    all_scores["magnitude"]
    .groupby("group")[score_cols]
    .mean()
    .loc[:, [cat_selected]]  # only the chosen category
    if cat_selected in score_cols else pd.DataFrame()
)
if not group_means_sel.empty:
    st.dataframe(group_means_sel.style.format("{:.4f}"))
else:
    st.info("Selected category has no computed composites yet (check top-k overlap with available features).")

# ============================================================
# FEATURE DISTRIBUTION VIEWER (smoothing control)
# ============================================================
st.subheader("Single Feature Viewer (smoothed distributions)")

valid_feature_options = sorted([f for f in features if f in merged_df.columns])

col_fd1, col_fd2, col_fd3, col_fd4 = st.columns([1.8, 1.2, 1.4, 1.2])
with col_fd1:
    fd_feature = st.selectbox(
        "Feature to visualize",
        valid_feature_options,
        key="fd_feature",
        help="Pick a single feature to view smoothed **group-wise** distributions."
    )
all_groups_fd = sorted(merged_df["group"].dropna().unique())
with col_fd2:
    fd_groups = st.multiselect(
        "Which groups to display?",
        all_groups_fd, default=all_groups_fd, key="fd_groups",
        help="Toggle which groups appear in the plot. You can also click legend entries to hide/show lines."
    )
with col_fd3:
    fd_tid = st.selectbox(
        "Reference individual (TID) to draw as vertical line",
        merged_df["TID"].sort_values().unique(),
        key="fd_tid",
        help="We annotate the dashed vertical line with that individual's **percentile** within CN and AD (if available)."
    )
with col_fd4:
    bw_mult = st.slider(
        "Smoothing (bandwidth × Scott’s rule)",
        0.3, 3.0, 1.0, 0.1,
        help="Lower values show more detail; higher values yield smoother curves."
    )

# Smoothed group densities for the selected feature
fig_fd = go.Figure()
vals_all = merged_df.loc[merged_df["group"].isin(fd_groups), fd_feature].dropna().values
if vals_all.size >= 2:
    lo, hi = np.percentile(vals_all, [1, 99])
    pad = 0.1 * (hi - lo if hi > lo else 1.0)
    xs = np.linspace(lo - pad, hi + pad, 256)
else:
    xs = np.array([])

ymax = 0.0
group_vals_map = {}  # store raw vals per group for percentile calc
for g in fd_groups:
    vals = merged_df.loc[merged_df["group"] == g, fd_feature].dropna().values
    group_vals_map[str(g).lower()] = vals
    if vals.size >= 2 and xs.size:
        kde = gaussian_kde(vals)
        # SciPy vs fallback compatibility
        try:
            kde.set_bandwidth(kde.factor * bw_mult)
        except TypeError:
            kde.set_bandwidth(bw_method=kde.factor * bw_mult)  # newer SciPy signature
        ys = kde(xs)
        ymax = max(ymax, float(np.nanmax(ys)))
        line_col, fill_col = color_for_group(g)
        fig_fd.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{g}",
            line=dict(color=line_col, width=2),
            fill="tozeroy", fillcolor=fill_col,
            hovertemplate=f"Group: {g}<br>{fd_feature}: %{{x:.4f}}<br>Density: %{{y:.2f}}<extra></extra>"
        ))
    elif vals.size == 1:
        line_col, _ = color_for_group(g)
        fig_fd.add_trace(go.Scatter(
            x=vals, y=[0], mode="markers", name=f"{g} (single)",
            marker=dict(color=line_col, size=8)
        ))

# Individual reference line + percentiles
indiv_val = merged_df.loc[merged_df["TID"] == fd_tid, fd_feature]
annot_text = None
if not indiv_val.empty and np.isfinite(indiv_val.iloc[0]):
    v = float(indiv_val.iloc[0])
    fig_fd.add_vline(x=v, line_width=2, line_dash="dash", line_color="black")
    # Percentiles wrt CN and AD (if present)
    p_cn = percentile_rank(group_vals_map.get("cn", []), v)
    p_ad = percentile_rank(group_vals_map.get("ad", []), v)
    parts = []
    if np.isfinite(p_cn):
        parts.append(f"CN: {p_cn:.1f}th perc.")
    if np.isfinite(p_ad):
        parts.append(f"AD: {p_ad:.1f}th perc.")
    if parts:
        annot_text = f"{fd_tid} @ {fd_feature}\n" + " • ".join(parts)
        fig_fd.add_annotation(
            x=v, y=(ymax * 0.95 if ymax > 0 else 0.0),
            text=annot_text, showarrow=False,
            align="left", bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)",
            font=PLOT_FONT
        )

fig_fd.update_layout(
    height=380,
    xaxis_title=fd_feature,
    yaxis_title="Density",
    template="plotly_white",
    legend_title="Group (click to toggle)",
    hovermode="x unified",
    font=PLOT_FONT,
    xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR),
    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR),
    margin=dict(l=20, r=20, t=40, b=40)
)
st.plotly_chart(fig_fd, use_container_width=True)
st.caption("Dashed vertical line marks the selected individual. The annotation shows their percentile within CN and AD, based on the displayed distributions.")

# Transcript under feature viewer
st.markdown("### Transcript for Selected Individual")
text_match = text_df[text_df["TID"] == fd_tid] if not text_df.empty else pd.DataFrame()
if not text_match.empty:
    st.text_area("Transcript", value=text_match.iloc[0]["text"], height=280)
else:
    st.info("Transcript not found (no transcript files uploaded or TID not present).")

# ============================================================
# Sidebar selections for composite distributions (selected category only)
# ============================================================
st.sidebar.header("Analysis Selection")
st.sidebar.markdown(
    "Use the controls below to choose **which person and groups** to visualize in the composite score plots. "
    "These settings affect only the **Composite Score** section."
)

groups = sorted(all_scores["magnitude"]["group"].dropna().unique())
selected_group = st.sidebar.selectbox(
    "A) Choose a base group to pick the individual from",
    groups,
    help="Select the cohort whose member you want to highlight in the composite plots."
)
tids = all_scores["magnitude"][all_scores["magnitude"]["group"] == selected_group]["TID"].sort_values()
tid_selected = st.sidebar.selectbox(
    "B) Pick the individual (TID) to highlight",
    tids,
    help="A dashed line will indicate this person's composite score in the distribution(s) below."
)
selected_groups = st.sidebar.multiselect(
    "C) Which groups should be shown in the composite distributions?",
    groups,
    default=groups,
    help="Toggle which groups appear in the composite score plots. You can also click legend entries to hide/show lines."
)

# === ONLY THE SELECTED CATEGORY BELOW ===
st.subheader(f"Composite Score Distributions — **{cat_selected}**")

mode_single_or_split = st.radio(
    "How should the selected category be displayed?",
    ["Single composite", "Split into two by sign (+/−)"],
    help=(
        "• **Single composite**: one |SHAP|-weighted (L1-normalized, if selected above) deviation score for this category.\n"
        "• **Split by sign**: two directional composites, using only features with positive SHAP (\"+\") or negative SHAP (\"−\")."
    ),
    horizontal=True,
)

# Keep per-category labels just for the selected category
if "pos_label_map" not in st.session_state:
    st.session_state.pos_label_map = {}
if "neg_label_map" not in st.session_state:
    st.session_state.neg_label_map = {}
st.session_state.pos_label_map.setdefault(cat_selected, "AD-like (+)")
st.session_state.neg_label_map.setdefault(cat_selected, "CN-like (−)")

with st.expander("Relabel (+/−) for this category & export CSV", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.pos_label_map[cat_selected] = st.text_input(
            f"{cat_selected} (+) label",
            value=st.session_state.pos_label_map[cat_selected],
            help="Rename the **positive-SHAP** subcategory label (e.g., 'AD-like')."
        )
    with c2:
        st.session_state.neg_label_map[cat_selected] = st.text_input(
            f"{cat_selected} (−) label",
            value=st.session_state.neg_label_map[cat_selected],
            help="Rename the **negative-SHAP** subcategory label (e.g., 'CN-like')."
        )

    # Build export dataframe for ALL participants but ONLY this category
    df_mag = all_scores["magnitude"]
    df_pos = all_scores["dir_ADlike"]
    df_neg = all_scores["dir_CNlike"]

    if mode_single_or_split == "Single composite":
        export_df = df_mag[["TID", "group", cat_selected]].rename(columns={cat_selected: f"{cat_selected} — Deviation"})
        file_name = f"{cat_selected}_single_k{k_selected}_l1{int(norm_l1)}.csv"
    else:
        export_df = df_pos[["TID", "group"]].copy()
        export_df[f"{cat_selected} — {st.session_state.pos_label_map[cat_selected]}"] = df_pos[cat_selected]
        export_df[f"{cat_selected} — {st.session_state.neg_label_map[cat_selected]}"] = df_neg[cat_selected]
        file_name = f"{cat_selected}_split_k{k_selected}_l1{int(norm_l1)}.csv"

    st.download_button(
        "Download CSV (selected category)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        help="Exports composites for the selected category for all participants, with current top-k/normalization and labels."
    )

# Helper to render a KDE panel (with shaded fill + percentile annotation)
def render_kde_panel(values_by_group, title, vline_value=None, height=300):
    fig = go.Figure()
    # common x grid
    all_vals = np.concatenate([v for v in values_by_group.values() if v.size > 0]) if values_by_group else np.array([])
    if all_vals.size >= 2:
        lo, hi = np.percentile(all_vals, [1, 99])
        pad = 0.1 * (hi - lo if hi > lo else 1.0)
        xs = np.linspace(lo - pad, hi + pad, 256)
    else:
        xs = np.array([])

    y_max = 0.0
    for g, vals in values_by_group.items():
        if vals.size >= 2 and xs.size:
            kde = gaussian_kde(vals)
            ys = kde(xs)
            y_max = max(y_max, float(np.nanmax(ys)))
            line_col, fill_col = color_for_group(g)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=f"{g}",
                line=dict(color=line_col, width=2),
                fill="tozeroy", fillcolor=fill_col,
                hovertemplate=f"Group: {g}<br>{title}: %{{x:.4f}}<br>Density: %{{y:.2f}}<extra></extra>"
            ))
        elif vals.size == 1:
            line_col, _ = color_for_group(g)
            fig.add_trace(go.Scatter(x=vals, y=[0], mode="markers", name=f"{g} (single)",
                                     marker=dict(color=line_col, size=8)))

    # Percentiles wrt CN and AD (if vline is given)
    if (vline_value is not None) and np.isfinite(vline_value):
        fig.add_vline(x=vline_value, line_width=2, line_dash="dash", line_color="black")
        vals_cn = values_by_group.get("cn", np.array([]))
        vals_ad = values_by_group.get("ad", np.array([]))
        p_cn = percentile_rank(vals_cn, vline_value) if vals_cn.size else np.nan
        p_ad = percentile_rank(vals_ad, vline_value) if vals_ad.size else np.nan
        parts = []
        if np.isfinite(p_cn): parts.append(f"CN: {p_cn:.1f}th perc.")
        if np.isfinite(p_ad): parts.append(f"AD: {p_ad:.1f}th perc.")
        if parts:
            fig.add_annotation(
                x=vline_value, y=(y_max * 0.95 if y_max > 0 else 0.0),
                text=" | ".join(parts), showarrow=False,
                align="center", bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)",
                font=PLOT_FONT
            )

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis_title=title,
        yaxis_title="Density",
        template="plotly_white",
        legend_title="Group (click to toggle)",
        hovermode="x unified",
        font=PLOT_FONT,
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# -------- Render ONLY the selected category --------
df_mag = all_scores["magnitude"]
df_pos = all_scores["dir_ADlike"]
df_neg = all_scores["dir_CNlike"]
meta_topk = all_scores["meta_topk"]
Z_all     = all_scores["Z_all"]

if mode_single_or_split == "Single composite":
    values_by_group = {str(g).lower(): df_mag.loc[df_mag["group"] == g, cat_selected].dropna().values for g in selected_groups}
    indiv_val = df_mag.loc[df_mag["TID"] == tid_selected, cat_selected]
    vline = float(indiv_val.iloc[0]) if not indiv_val.empty else None

    render_kde_panel(values_by_group,
                     title=f"{cat_selected} — Deviation (|SHAP|, L1-norm={norm_l1})",
                     vline_value=vline)

    # --- Details (included features & formula) ---
    with st.expander(f"Details: {cat_selected} (features & formula)"):
        meta_cat = (
            meta_topk[(meta_topk["category"] == cat_selected) & (meta_topk["Feature"].isin(merged_df.columns))]
            .sort_values("abs_shap", ascending=False)
            .head(k_selected)
            .copy()
        )
        meta_cat["Sign"] = np.where(meta_cat["Mean SHAP value"] >= 0, "+", "−")
        st.markdown("**Included features (top-k within this category)**")
        st.dataframe(
            meta_cat[["Feature", "Mean SHAP value", "abs_shap", "Sign"]]
            .rename(columns={"abs_shap": "Mean |SHAP value|"}),
            use_container_width=True
        )

        feats = meta_cat["Feature"].tolist()
        shap_vals = meta_cat.set_index("Feature")["Mean SHAP value"]

        # z-values for the selected individual
        if feats:
            z_series = Z_all.loc[merged_df["TID"] == tid_selected, feats]
            z_row = (z_series.iloc[0].to_dict() if not z_series.empty else {f: np.nan for f in feats})
        else:
            z_row = {}

        st.markdown("**Computation (Deviation) for selected individual**")
        terms, num, den = [], 0.0, 0.0
        for f in feats:
            s_abs = abs(float(shap_vals[f]))
            z = float(z_row.get(f, np.nan))
            if np.isfinite(z):
                terms.append(f"{s_abs:.4f}·z({f})")
                num += s_abs * z
                den += s_abs
        if norm_l1:
            st.latex(r"\mathrm{Score}_{i," + cat_selected.replace('_', r'\_') + r"}=\frac{\sum |s_f|\,z_{i,f}}{\sum |s_f|}")
            st.markdown(f"`Expanded:`  ({' + '.join(terms) if terms else '0'})  /  {den:.4f}  =  **{(num/den if den else float('nan')):.4f}**")
        else:
            st.latex(r"\mathrm{Score}_{i," + cat_selected.replace('_', r'\_') + r"}=\sum |s_f|\,z_{i,f}")
            st.markdown(f"`Expanded:`  {' + '.join(terms) if terms else '0'}  =  **{num:.4f}**")

else:
    # SPLIT into two by sign: (+) uses s>0 ; (−) uses s<0
    pos_title = f"{cat_selected} — {st.session_state.pos_label_map[cat_selected]}"
    neg_title = f"{cat_selected} — {st.session_state.neg_label_map[cat_selected]}"

    # (+) panel
    vals_pos = {str(g).lower(): df_pos.loc[df_pos["group"] == g, cat_selected].dropna().values for g in selected_groups}
    indiv_pos = df_pos.loc[df_pos["TID"] == tid_selected, cat_selected]
    vline_pos = float(indiv_pos.iloc[0]) if not indiv_pos.empty else None
    render_kde_panel(vals_pos, title=pos_title, vline_value=vline_pos)

    # (−) panel
    vals_neg = {str(g).lower(): df_neg.loc[df_neg["group"] == g, cat_selected].dropna().values for g in selected_groups}
    indiv_neg = df_neg.loc[df_neg["TID"] == tid_selected, cat_selected]
    vline_neg = float(indiv_neg.iloc[0]) if not indiv_neg.empty else None
    render_kde_panel(vals_neg, title=neg_title, vline_value=vline_neg)

    # --- Details: two separate expanders with sign-filtered features & formulas ---
    meta_cat_all = (
        meta_topk[(meta_topk["category"] == cat_selected) & (meta_topk["Feature"].isin(merged_df.columns))]
        .sort_values("abs_shap", ascending=False)
        .head(k_selected)
        .copy()
    )
    meta_cat_all["SignVal"] = meta_cat_all["Mean SHAP value"]
    meta_cat_all["Sign"] = np.where(meta_cat_all["SignVal"] >= 0, "+", "−")

    # (+) expander
    with st.expander(f"Details: {pos_title} (s_f>0)"):
        meta_pos = meta_cat_all[meta_cat_all["SignVal"] > 0].copy()
        st.dataframe(
            meta_pos[["Feature", "Mean SHAP value", "abs_shap", "Sign"]]
            .rename(columns={"abs_shap": "Mean |SHAP value|"}),
            use_container_width=True
        )
        feats = meta_pos["Feature"].tolist()
        shap_vals = meta_pos.set_index("Feature")["Mean SHAP value"]
        if feats:
            z_series = Z_all.loc[merged_df["TID"] == tid_selected, feats]
            z_row = (z_series.iloc[0].to_dict() if not z_series.empty else {f: np.nan for f in feats})
        else:
            z_row = {}
        st.markdown("**Computation (AD-like, signed)**")
        pos_terms, num_ad, den_ad = [], 0.0, 0.0
        for f in feats:
            s_f = float(shap_vals[f]); z = float(z_row.get(f, np.nan))
            if np.isfinite(z):
                pos_terms.append(f"{s_f:.4f}·z({f})")
                num_ad += s_f * z; den_ad += abs(s_f)
        if norm_l1:
            st.latex(r"\mathrm{ADlike}_{i," + cat_selected.replace('_', r'\_') + r"}=\frac{\sum_{s_f>0} s_f\,z_{i,f}}{\sum_{s_f>0}|s_f|}")
            st.markdown(f"`Expanded:`  ({' + '.join(pos_terms) if pos_terms else '0'}) / {den_ad:.4f} = **{(num_ad/den_ad if den_ad else float('nan')):.4f}**")
        else:
            st.latex(r"\mathrm{ADlike}_{i," + cat_selected.replace('_', r'\_') + r"}=\sum_{s_f>0} s_f\,z_{i,f}")
            st.markdown(f"`Expanded:`  {' + '.join(pos_terms) if pos_terms else '0'} = **{num_ad:.4f}**")

    # (−) expander
    with st.expander(f"Details: {neg_title} (s_f<0)"):
        meta_neg = meta_cat_all[meta_cat_all["SignVal"] < 0].copy()
        st.dataframe(
            meta_neg[["Feature", "Mean SHAP value", "abs_shap", "Sign"]]
            .rename(columns={"abs_shap": "Mean |SHAP value|"}),
            use_container_width=True
        )
        feats = meta_neg["Feature"].tolist()
        shap_vals = meta_neg.set_index("Feature")["Mean SHAP value"]
        if feats:
            z_series = Z_all.loc[merged_df["TID"] == tid_selected, feats]
            z_row = (z_series.iloc[0].to_dict() if not z_series.empty else {f: np.nan for f in feats})
        else:
            z_row = {}
        st.markdown("**Computation (CN-like, signed; positive magnitude)**")
        neg_terms, num_cn, den_cn = [], 0.0, 0.0
        for f in feats:
            s_f = float(shap_vals[f]); z = float(z_row.get(f, np.nan))
            if np.isfinite(z):
                neg_terms.append(f"{-s_f:.4f}·(-z({f}))")
                num_cn += (-s_f) * (-z)   # equals - s_f * z
                den_cn += abs(s_f)
        if norm_l1:
            st.latex(r"\mathrm{CNlike}_{i," + cat_selected.replace('_', r'\_') + r"}=\frac{-\sum_{s_f<0} s_f\,z_{i,f}}{\sum_{s_f<0}|s_f|}")
            st.markdown(f"`Expanded:`  ({' + '.join(neg_terms) if neg_terms else '0'}) / {den_cn:.4f} = **{(num_cn/den_cn if den_cn else float('nan')):.4f}**")
        else:
            st.latex(r"\mathrm{CNlike}_{i," + cat_selected.replace('_', r'\_') + r"}=-\sum_{s_f<0} s_f\,z_{i,f}")
            st.markdown(f"`Expanded:`  {' + '.join(neg_terms) if neg_terms else '0'} = **{num_cn:.4f}**")
