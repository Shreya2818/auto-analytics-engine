"""
Auto-Analytics Engine — Streamlit dashboard
Run from the project folder with: streamlit run app.py

This file handles the interface only. Analysis logic remains in engine.py.
Expected default dataset: data/districts.csv
"""

from __future__ import annotations

import html
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import engine as E


BASE = Path(__file__).parent
DEFAULT_CSV = BASE / "data" / "districts.csv"
OUT_DIR = BASE / "outputs"

SEV_COLORS = {
    "High": "#fb7185",
    "Medium": "#fbbf24",
    "Low": "#34d399",
}
METHOD_LABELS = {
    "iqr": "IQR rule",
    "zscore": "Z-score",
    "mad": "Modified Z-score (MAD)",
}

st.set_page_config(
    page_title="Auto-Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_ui_theme() -> None:
    """Apply a cohesive dark dashboard theme."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --panel: #111a2e;
            --line: rgba(148, 163, 184, 0.18);
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #8b5cf6;
            --accent2: #38bdf8;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 0%, rgba(139,92,246,.17), transparent 32%),
                radial-gradient(circle at 92% 4%, rgba(56,189,248,.10), transparent 26%),
                linear-gradient(180deg, #0b1020 0%, #0a0f1c 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1500px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(13, 20, 37, .96);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: -.02em;
        }

        .hero-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1.2rem;
            padding: 1.4rem 1.5rem;
            margin: .3rem 0 1rem 0;
            border: 1px solid rgba(167,139,250,.25);
            border-radius: 20px;
            background: linear-gradient(120deg, rgba(139,92,246,.19), rgba(56,189,248,.07) 70%, rgba(17,26,46,.8));
            box-shadow: 0 14px 38px rgba(0,0,0,.18);
        }

        .hero-copy { display: flex; flex-direction: column; gap: .35rem; }
        .eyebrow {
            color: #c4b5fd;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
        }
        .hero-banner h1 {
            color: #f8fafc;
            font-size: clamp(1.8rem, 3vw, 2.7rem);
            line-height: 1.1;
            margin: 0;
            letter-spacing: -.045em;
        }
        .hero-subtitle { color: #cbd5e1; font-size: .95rem; margin-top: .3rem; }

        .meta-badges {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: .5rem;
        }
        .meta-badge {
            color: #e2e8f0;
            background: rgba(15,23,42,.62);
            border: 1px solid rgba(196,181,253,.25);
            border-radius: 999px;
            padding: .45rem .75rem;
            font-size: .78rem;
            font-weight: 650;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(24,35,59,.96), rgba(15,23,42,.94));
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 16px;
            padding: .75rem .9rem;
            box-shadow: 0 10px 26px rgba(0,0,0,.13);
            transition: transform .18s ease, border-color .18s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(167,139,250,.58);
        }
        div[data-testid="stMetricLabel"] { color: #cbd5e1; font-weight: 600; }
        div[data-testid="stMetricValue"] { color: #f8fafc; font-size: 1.9rem !important; font-weight: 750; }

        .stTabs [role="tablist"] { gap: .45rem; border-bottom: 1px solid var(--line); }
        .stTabs [role="tab"] {
            border: 1px solid transparent;
            border-radius: 10px 10px 0 0;
            padding: .6rem .9rem;
            color: #cbd5e1;
            font-weight: 650;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            color: #fff;
            background: rgba(139,92,246,.17);
            border-color: rgba(167,139,250,.35);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(17,26,46,.7);
            border: 1px dashed rgba(167,139,250,.45);
            border-radius: 14px;
        }

        .stButton > button, .stDownloadButton > button {
            border: 1px solid rgba(167,139,250,.3);
            border-radius: 10px;
            background: linear-gradient(135deg, #8b5cf6, #6d28d9);
            color: #fff;
            font-weight: 700;
            min-height: 40px;
            transition: transform .18s ease, box-shadow .18s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            color: #fff;
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(109,40,217,.3);
        }

        [data-testid="stPlotlyChart"] {
            border: 1px solid rgba(148,163,184,.13);
            border-radius: 14px;
            padding: .3rem;
            background: rgba(15,23,42,.3);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,.14);
            border-radius: 12px;
            overflow: hidden;
        }
        .stAlert { border-radius: 12px; }
        .section-note { color: #94a3b8; font-size: .88rem; }
        @media (max-width: 768px) {
            .hero-banner { align-items: flex-start; padding: 1.1rem; }
            .meta-badges { justify-content: flex-start; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=None):
    """Give all Plotly charts a consistent dark appearance."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#e5e7eb", size=12),
        title_font=dict(size=17, color="#f8fafc"),
        margin=dict(l=20, r=20, t=58, b=25),
        legend=dict(bgcolor="rgba(15,23,42,.35)", borderwidth=0),
        hoverlabel=dict(bgcolor="#172033", font_color="#f8fafc"),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def safe_text(value) -> str:
    """Escape dynamic text before inserting it into HTML."""
    return html.escape(str(value))


def display_dataframe(data, *args, **kwargs):
    """
    Display a DataFrame safely in Streamlit.

    Streamlit converts tables through PyArrow. Mixed values in object columns
    (for example, integers and strings in the entity column) can cause
    ArrowTypeError. Normalize object columns in a display-only copy so the
    underlying analysis data and calculations remain unchanged.
    """
    if isinstance(data, pd.io.formats.style.Styler):
        frame = data.data.copy()
        # Preserve number formatting where possible; normalize only object columns.
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        st.dataframe(data, *args, **kwargs)
        return

    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].map(
                lambda value: str(value) if pd.notna(value) else None
            )

    # Display a normalized frame. Using a plain frame avoids Arrow conversion
    # failures from Styler objects whose underlying columns contain mixed types.
    st.dataframe(frame, *args, **kwargs)


@st.cache_data(show_spinner=False)
def load_cached(raw: bytes) -> E.Dataset:
    """Parse the CSV once per distinct file."""
    return E.load_data(io.BytesIO(raw))


inject_ui_theme()

# --------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 📊 AutoAnalytics")
st.sidebar.caption("Automated insight generation")
st.sidebar.divider()

st.sidebar.subheader("01 · Dataset")
upload = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if upload is not None:
    raw, source_name = upload.getvalue(), upload.name
elif DEFAULT_CSV.exists():
    raw, source_name = DEFAULT_CSV.read_bytes(), DEFAULT_CSV.name
else:
    st.error("No dataset found. Upload a CSV or place your file at `data/districts.csv`.")
    st.stop()

try:
    ds = load_cached(raw)
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Could not load this dataset: {exc}")
    st.info("Check that the CSV contains a period column, an entity column, and numeric indicator columns.")
    st.stop()

st.sidebar.subheader("02 · Filters")
entities = sorted(ds.df["entity"].dropna().astype(str).unique().tolist())
periods_all = list(ds.period_order)
indicators_all = list(ds.indicators)

sel_entities = st.sidebar.multiselect(
    ds.entity_col.title(), entities, default=entities
)
sel_periods = st.sidebar.multiselect(
    ds.period_col.replace("_", " ").title(), periods_all, default=periods_all
)
sel_inds = st.sidebar.multiselect(
    "Indicators", indicators_all, default=indicators_all
)

st.sidebar.subheader("03 · Detection settings")
trend_thr = st.sidebar.slider("Significant trend change (%)", 1, 100, 10)
method = st.sidebar.selectbox(
    "Outlier detection method",
    list(METHOD_LABELS),
    format_func=METHOD_LABELS.get,
)
out_thr = st.sidebar.slider(
    "Outlier threshold",
    0.5,
    6.0,
    float(E.DEFAULT_OUTLIER_THRESHOLD[method]),
    0.1,
    key=f"out_thr_{method}",
    help="IQR: fence multiplier; Z-score and MAD: score threshold.",
)
by_period = st.sidebar.checkbox(
    f"Compare outliers with peers in the same {ds.period_word}",
    value=E.resolve_by_period(ds, E.Config()),
)
corr_thr = st.sidebar.slider("Correlation minimum |r|", 0.30, 0.99, 0.70, 0.01)

with st.sidebar.expander("Severity settings"):
    st.caption("Severity depends on signal strength relative to the threshold.")
    medium_mult = st.slider("Medium at (× threshold)", 1.0, 3.0, 1.5, 0.05)
    high_mult = st.slider("High at (× threshold)", 1.0, 5.0, 2.0, 0.05)
    if medium_mult > high_mult:
        st.warning("High severity cutoff must be at least the Medium cutoff.")
        high_mult = medium_mult

with st.sidebar.expander("Indicator direction"):
    st.caption("Choose whether a higher or lower value is preferable.")
    directions = {}
    for ind in ds.indicators:
        guess = E.infer_direction(ind)
        directions[ind] = st.selectbox(
            ind,
            ["higher_better", "lower_better"],
            index=["higher_better", "lower_better"].index(guess),
            key=f"dir_{ind}",
            format_func=lambda x: "Higher is better" if x == "higher_better" else "Lower is better",
        )

with st.sidebar.expander("Target thresholds"):
    st.caption("Set a minimum or maximum to detect threshold breaches. A value of 0 disables the target.")
    targets = {}
    for ind in ds.indicators:
        left, right = st.columns([1.1, 1])
        kind = left.selectbox("Rule", ["none", "min", "max"], key=f"tk_{ind}", label_visibility="collapsed")
        val = right.number_input("Value", value=0.0, key=f"tv_{ind}", label_visibility="collapsed")
        if kind != "none" and val != 0:
            targets[ind] = (kind, val)

with st.sidebar.expander("Data quality settings"):
    inferred = [
        ind for ind in ds.indicators
        if E.is_pct_like(ind, ds.df[ind], E.Config())
    ]
    pct_cols = st.multiselect(
        "Indicators expected to be between 0 and 100",
        ds.indicators,
        default=inferred,
    )

cfg = E.Config(
    trend_threshold=float(trend_thr),
    outlier_method=method,
    outlier_threshold=float(out_thr),
    outlier_by_period=by_period,
    corr_threshold=float(corr_thr),
    medium_mult=medium_mult,
    high_mult=high_mult,
    targets=targets,
    directions=directions,
    pct_columns=pct_cols,
)

# --------------------------------------------------------------------- header
st.markdown(
    f"""
    <div class="hero-banner">
      <div class="hero-copy">
        <span class="eyebrow">Operational intelligence · Automated analytics</span>
        <h1>Auto-Analytics Engine</h1>
        <div class="hero-subtitle">Turn raw data into clear trends, anomalies, and actionable findings.</div>
      </div>
      <div class="meta-badges">
        <span class="meta-badge">📁 {safe_text(source_name)}</span>
        <span class="meta-badge">▦ {len(ds.df):,} rows</span>
        <span class="meta-badge">◉ {ds.df['entity'].nunique()} {safe_text(ds.entity_col)}s</span>
        <span class="meta-badge">◷ {len(ds.period_order)} periods</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if ds.period_order:
    st.caption(f"Reporting window: **{ds.period_order[0]}** to **{ds.period_order[-1]}** · Source: `{source_name}`")

if not (sel_entities and sel_periods and sel_inds):
    st.warning("Choose at least one entity, one period, and one indicator in the sidebar.")
    st.stop()

# --------------------------------------------------------------------- analysis
with st.spinner("Analyzing your dataset…"):
    an = E.run_analysis(ds, cfg, E.Filters(sel_entities, sel_periods, sel_inds))

ins = an.insights.copy()

for warning in ds.report.get("warnings", []):
    st.warning(warning)
for note in an.notes:
    st.info(note)

# Robust entity count: missing entity names must not break the boolean mask.
if not ins.empty and "entity" in ins.columns:
    entity_names = ins["entity"].fillna("").astype(str)
    flagged = entity_names[~entity_names.str.startswith("ALL ")].replace("", pd.NA).nunique()
else:
    flagged = 0

counts_df = E.severity_counts(ins)
counts = counts_df.set_index("severity")["count"] if not counts_df.empty else pd.Series(dtype=int)
high_count = int(counts.get("High", 0))
medium_count = int(counts.get("Medium", 0))
low_count = int(counts.get("Low", 0))

st.markdown("### Dashboard overview")
st.caption("A live summary of the findings produced by your current filters and thresholds.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total insights", f"{len(ins):,}", help="Total findings returned by the analysis engine.")
m2.metric("High severity", f"{high_count:,}", help="Findings requiring the closest attention.")
m3.metric("Medium / Low", f"{medium_count:,} / {low_count:,}")
m4.metric(f"{ds.entity_col.title()}s flagged", f"{flagged:,}")

tab_ins, tab_sev, tab_cor, tab_trend, tab_dq = st.tabs(
    ["✨ Insights", "🚦 Severity", "🔗 Correlation", f"📈 Trends by {ds.entity_col}", "🧹 Data quality"]
)

# ---------------------------------------------------------------------- insights
with tab_ins:
    st.markdown("### Key findings")
    f1, f2 = st.columns(2)
    types = f1.multiselect(
        "Insight type",
        ["trend", "outlier", "correlation", "threshold_breach"],
        default=["trend", "outlier", "correlation", "threshold_breach"],
    )
    sevs = f2.multiselect("Severity", E.LEVELS, default=E.LEVELS)

    if ins.empty:
        view = ins
    else:
        view = ins[ins["type"].isin(types) & ins["severity"].isin(sevs)].copy()

    if view.empty:
        st.info("No insights match the current filters. Try changing the selected types, severity, or thresholds.")
    else:
        st.caption(f"{len(view):,} matching insights · Showing the first {min(8, len(view))} highlights")
        for r in view.head(8).itertuples():
            severity = str(getattr(r, "severity", "Unknown"))
            color = SEV_COLORS.get(severity, "#94a3b8")
            insight_id = safe_text(getattr(r, "insight_id", "—"))
            explanation = safe_text(getattr(r, "explanation", "No explanation provided"))
            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,rgba(24,35,59,.96),rgba(15,23,42,.95));
                    border:1px solid rgba(148,163,184,.17);
                    border-left:4px solid {color};
                    border-radius:14px;
                    padding:16px 18px;
                    margin:10px 0;
                ">
                  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:9px;">
                    <span style="color:#94a3b8;font-size:12px;font-weight:600;">ID · {insight_id}</span>
                    <span style="background:{color}22;color:{color};border:1px solid {color}66;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:800;letter-spacing:.06em;">{safe_text(severity.upper())}</span>
                  </div>
                  <div style="color:#f1f5f9;font-size:14px;line-height:1.7;">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def paint_severity(value):
            color = SEV_COLORS.get(value, "#94a3b8")
            return f"background-color: {color}; color: #0b1020; font-weight: 700"

        st.markdown("#### Full insight table")
        styled_view = view.style
        if "severity" in view.columns:
            styled_view = styled_view.map(paint_severity, subset=["severity"])
        display_dataframe(
            styled_view.format(na_rep="—", precision=2),
            width="stretch",
            hide_index=True,
            height=420,
        )

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "⬇ Download insights CSV",
        view.to_csv(index=False),
        "insights.csv",
        "text/csv",
        disabled=view.empty,
        use_container_width=True,
    )
    d2.download_button(
        "⬇ Download insights JSON",
        E.insights_to_json(view),
        "insights.json",
        "application/json",
        disabled=view.empty,
        use_container_width=True,
    )
    if d3.button("💾 Save outputs to ./outputs/", use_container_width=True):
        OUT_DIR.mkdir(exist_ok=True)
        view.to_csv(OUT_DIR / "insights.csv", index=False)
        (OUT_DIR / "insights.json").write_text(E.insights_to_json(view), encoding="utf-8")
        an.corr.to_csv(OUT_DIR / "correlation_matrix.csv")
        st.success(f"Saved insights.csv, insights.json, and correlation_matrix.csv to {OUT_DIR}")

# ---------------------------------------------------------------------- severity
with tab_sev:
    st.markdown("### Severity distribution")
    st.caption("Review how findings are distributed across severity levels.")
    c1, c2 = st.columns(2)

    sc = E.severity_counts(ins)
    fig = px.bar(
        sc,
        x="severity",
        y="count",
        color="severity",
        text="count",
        color_discrete_map=SEV_COLORS,
        title="Insights by severity",
        category_orders={"severity": E.LEVELS},
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Number of insights")
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    c1.plotly_chart(style_chart(fig, 360), width="stretch", key="severity_distribution")

    if not ins.empty:
        bt = ins.groupby(["type", "severity"]).size().reset_index(name="count")
        fig2 = px.bar(
            bt,
            x="type",
            y="count",
            color="severity",
            barmode="stack",
            color_discrete_map=SEV_COLORS,
            title="Severity by insight type",
            category_orders={"severity": E.LEVELS},
        )
        fig2.update_layout(xaxis_title=None, yaxis_title="Number of insights")
        c2.plotly_chart(style_chart(fig2, 360), width="stretch", key="severity_by_type")
    else:
        c2.info("No insights available for the selected filters.")

# ------------------------------------------------------------------- correlation
with tab_cor:
    st.markdown("### Indicator relationships")
    n_rows = int(
        (ds.df["entity"].isin(sel_entities) & ds.df["period"].isin(sel_periods)).sum()
    )
    st.warning(
        f"Pearson correlation is computed on n = {n_rows} rows. With few rows—and repeated observations from the same entity—correlations can be unstable. Treat them as indicative, not proof of causation."
    )

    if an.corr.empty:
        st.info("Select at least two indicators to see correlations.")
    else:
        fig = px.imshow(
            an.corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Pearson correlation matrix",
        )
        st.plotly_chart(style_chart(fig, max(420, 34 * len(an.corr))), width="stretch", key="correlation_heatmap")
        st.markdown(f"#### Strong correlation pairs · |r| ≥ {corr_thr:.2f}")
        if hasattr(an, "corr_pairs") and not an.corr_pairs.empty:
            display_dataframe(
                an.corr_pairs.style.format({"r": "{:.2f}", "p_value": "{:.3f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No pairs crossed the selected correlation threshold.")
        st.download_button(
            "⬇ Download correlation matrix CSV",
            an.corr.to_csv(),
            "correlation_matrix.csv",
            "text/csv",
        )

# ------------------------------------------------------------------------- trends
with tab_trend:
    st.markdown("### Trend explorer")
    if not sel_entities or not sel_periods or not sel_inds:
        st.info("Select entities, periods, and indicators in the sidebar.")
    else:
        ent = st.selectbox(ds.entity_col.title(), sel_entities)
        periods = [p for p in ds.period_order if p in sel_periods]
        inds = list(sel_inds)

        cols = 2 if len(inds) > 1 else 1
        rows = -(-len(inds) // cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[E.pretty(i) for i in inds],
            vertical_spacing=min(0.14, 0.8 / max(rows, 1)),
        )

        one = ds.df[
            (ds.df["entity"] == ent) & ds.df["period"].isin(periods)
        ].set_index("period")
        everyone = ds.df[ds.df["period"].isin(periods)].groupby("period")
        flags = an.trends[
            (an.trends["entity"] == ent) & an.trends["is_significant"]
        ]

        for k, ind in enumerate(inds):
            row, col = k // cols + 1, k % cols + 1
            med = everyone[ind].median().reindex(periods)
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=med.values,
                    mode="lines",
                    name=f"All-{ds.entity_col} median",
                    line=dict(dash="dot", color="#94a3b8", width=2),
                    showlegend=(k == 0),
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=one[ind].reindex(periods).values,
                    mode="lines+markers",
                    name=ent,
                    line=dict(color="#a78bfa", width=3),
                    marker=dict(size=7),
                    showlegend=(k == 0),
                ),
                row=row,
                col=col,
            )
            fl = flags[flags["indicator"] == ind]
            if not fl.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fl["period"],
                        y=fl["value"],
                        mode="markers",
                        name="Significant change",
                        marker=dict(color="#fb7185", size=12, symbol="diamond"),
                        text=[f"{v:+.1f}% vs previous" for v in fl["change_pct"]],
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=(k == 0),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            height=max(340, 310 * rows),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(style_chart(fig), width="stretch", key="trend_explorer")

        if not flags.empty:
            st.markdown(f"#### Significant changes in {ent}")
            st.caption(f"Threshold: absolute change ≥ {trend_thr}%")
            trend_cols = [
                c for c in [
                    "indicator", "period", "prev_value", "value",
                    "change_pct", "status", "severity"
                ] if c in flags.columns
            ]
            display_dataframe(flags[trend_cols], width="stretch", hide_index=True)
        else:
            st.info(f"No significant changes were detected for {ent} with the current threshold.")

# -------------------------------------------------------------------- data quality
with tab_dq:
    st.markdown("### Data quality report")
    rep = ds.report

    q1, q2, q3 = st.columns(3)
    q1.metric("Rows loaded", f"{len(ds.df):,}")
    q2.metric("Indicators", f"{len(ds.indicators):,}")
    missing_total = int(ds.df.isna().sum().sum())
    q3.metric("Missing cells", f"{missing_total:,}")

    with st.expander("Preview of the dataset", expanded=True):
        display_dataframe(rep["head"], width="stretch", hide_index=True)

    with st.expander("DataFrame info"):
        st.code(rep["info"])

    st.markdown("#### Missing values by column")
    missing = rep["missing"]
    if isinstance(missing, pd.DataFrame):
        display_dataframe(missing, width="stretch")
    else:
        display_dataframe(missing.to_frame(name="missing values"), width="stretch")

    st.markdown("#### Values outside the expected 0–100% range")
    if an.quality.empty:
        st.success("No percentage indicator exceeds its configured logical range.")
    else:
        display_dataframe(an.quality, width="stretch", hide_index=True)
        st.caption(
            "Coverage above 100% may indicate that numerator and denominator use different reporting populations. Validate the source definitions before acting on these values."
        )

st.divider()
st.caption("Auto-Analytics Engine · Findings are statistical signals; validate important conclusions against source data and domain knowledge.")
