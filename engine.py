"""
engine.py - general-purpose Auto-Analytics Engine (no UI code, no hardcoded
District / indicator names, no hardcoded narratives).
This module implements the data loading, validation, trend detection, outlier
analysis, correlation checks, threshold breach checks, and insight-generation
logic used by the Streamlit dashboard.
"""
from __future__ import annotations
import io
import json
import re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
LEVELS = ["Low", "Medium", "High"]
INSIGHT_COLUMNS = [
    "insight_id", "type", "indicator", "entity", "period", "metric",
    "value", "prev_value", "change", "change_pct", "severity", "explanation",
]
PERIOD_ALIASES = ["month", "period", "fiscal_year", "year", "date", "quarter"]
ENTITY_ALIASES = ["district", "entity", "region", "block", "facility", "state"]
DEFAULT_OUTLIER_THRESHOLD = {"iqr": 1.5, "zscore": 3.0, "mad": 3.5}
OUTLIER_METHODS = list(DEFAULT_OUTLIER_THRESHOLD)
_LOWER_BETTER_HINTS = (
    "risk", "death", "mortality", "cases", "complication",
    "incidence", "morbidity", "stillbirth",
)
_PCT_HINTS = ("coverage", "delivery", "immuni", "percent", "pct", "share", "proportion", "ratio", "rate")
_NOT_PCT_HINTS = ("count", "cases", "number", "total")
@dataclass
class Config:
    trend_threshold: float = 10.0
    outlier_method: str = "iqr"
    outlier_threshold: Optional[float] = None
    outlier_by_period: Optional[bool] = None
    corr_threshold: float = 0.70
    min_corr_n: int = 10
    medium_mult: float = 1.5
    high_mult: float = 2.0
    targets: dict = field(default_factory=dict)
    directions: dict = field(default_factory=dict)
    pct_columns: Optional[list] = None
    include_compound: bool = True
    include_quality: bool = True
    def outlier_thr(self) -> float:
        return self.outlier_threshold if self.outlier_threshold is not None else DEFAULT_OUTLIER_THRESHOLD[self.outlier_method]
    def validate(self) -> None:
        if self.outlier_method not in OUTLIER_METHODS:
            raise ValueError(f"outlier_method must be one of {OUTLIER_METHODS}")
        if self.medium_mult > self.high_mult:
            raise ValueError("medium_mult must be <= high_mult")
        if self.trend_threshold <= 0:
            raise ValueError("trend_threshold must be > 0")
@dataclass
class Filters:
    entities: Optional[list] = None
    periods: Optional[list] = None
    indicators: Optional[list] = None
@dataclass
class Dataset:
    df: pd.DataFrame
    entity_col: str
    period_col: str
    indicators: list
    period_order: list
    report: dict
    @property
    def period_word(self) -> str:
        return "month" if self.period_col.lower() == "month" else (
            "year" if "year" in self.period_col.lower() else "period"
        )
@dataclass
class Analysis:
    trends: pd.DataFrame
    outliers: pd.DataFrame
    breaches: pd.DataFrame
    corr: pd.DataFrame
    corr_pairs: pd.DataFrame
    quality: pd.DataFrame
    insights: pd.DataFrame
    notes: list
# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def fmt(x) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "n/a"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.0f}" if x.is_integer() else f"{x:.1f}"
def pretty(name: str) -> str:
    parts = re.split(r"[_\s]+", str(name).strip())
    out = [p.upper() if len(p) <= 3 else p for p in parts if p]
    s = " ".join(out)
    return s[0].upper() + s[1:] if s and not s[:3].isupper() else s
def infer_direction(name: str) -> str:
    n = str(name).lower()
    return "lower_better" if any(h in n for h in _LOWER_BETTER_HINTS) else "higher_better"
def get_direction(name: str, cfg: Config) -> str:
    return cfg.directions.get(name) or infer_direction(name)
def is_pct_like(name: str, s: pd.Series, cfg: Config) -> bool:
    if cfg.pct_columns is not None:
        return name in cfg.pct_columns
    n = str(name).lower()
    if any(h in n for h in _NOT_PCT_HINTS):
        return False
    s = s.dropna()
    if s.empty:
        return False
    return any(h in n for h in _PCT_HINTS) and s.median() <= 120 and s.max() <= 200
def _level_index(ratio: np.ndarray, cfg: Config) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=float)
    return (ratio >= cfg.medium_mult).astype(int) + (ratio >= cfg.high_mult).astype(int)
def _severity(ratio, worse, cfg: Config):
    idx = _level_index(ratio, cfg) + np.asarray(worse, dtype=int)
    return np.array(LEVELS)[np.minimum(idx, 2)]
def resolve_by_period(ds: Dataset, cfg: Config) -> bool:
    if cfg.outlier_by_period is not None:
        return cfg.outlier_by_period
    return ds.df["entity"].nunique() > 1
def severity_counts(insights: pd.DataFrame) -> pd.DataFrame:
    if insights.empty:
        return pd.DataFrame({"severity": LEVELS, "count": [0, 0, 0]})
    counts = insights["severity"].value_counts().reindex(LEVELS, fill_value=0)
    return pd.DataFrame({"severity": LEVELS, "count": counts.astype(int).to_list()})
def insights_to_json(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), indent=2)
# --------------------------------------------------------------------------- #
# Part A - loading & validation
# --------------------------------------------------------------------------- #
def load_data(source) -> Dataset:
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - input-boundary behavior
        raise ValueError(f"Could not read CSV: {exc}") from exc
    if raw.empty:
        raise ValueError("The CSV has no rows.")
    raw.columns = [str(c).strip() for c in raw.columns]
    lower = {c.lower(): c for c in raw.columns}
    period_col = next((lower[a] for a in PERIOD_ALIASES if a in lower), None)
    entity_col = next((lower[a] for a in ENTITY_ALIASES if a in lower), None)
    if period_col is None:
        raise ValueError(f"No time column found. Expected one of {PERIOD_ALIASES}.")
    if entity_col is None:
        raise ValueError(f"No entity column found. Expected one of {ENTITY_ALIASES}.")
    warnings: list = []
    df = raw.rename(columns={entity_col: "entity", period_col: "period"}).copy()
    df["entity"] = df["entity"].astype(str).str.strip()
    df["period"] = df["period"].astype(str).str.strip()
    indicators = []
    for c in df.columns:
        if c in ("entity", "period"):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() == 0:
            warnings.append(f"Column '{c}' has no numeric values and was ignored.")
            continue
        bad = int((df[c].notna() & ~num.notna()).sum())
        if bad:
            warnings.append(f"Column '{c}' had {bad} non-numeric values; they were converted to missing.")
        indicators.append(c)
    if not indicators:
        raise ValueError("No numeric indicator columns were found.")
    df = df[["entity", "period", *indicators]].copy()
    df = df.drop_duplicates(subset=["entity", "period"], keep="last").sort_values(["entity", "period"]).reset_index(drop=True)
    # A single observation cannot support a within-entity trend.
    periods_per_entity = df.groupby("entity")["period"].nunique()
    for entity, n_periods in periods_per_entity.items():
        if n_periods == 1:
            warnings.append(
                f"{entity} has only one period; trend analysis for this entity is skipped."
            )
    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    report = {
        "head": df.head().copy(),
        "info": info_buffer.getvalue(),
        "missing": df.isna().sum().to_frame(name="missing"),
        "warnings": warnings,
    }
    period_order = list(dict.fromkeys(df["period"].tolist()))
    return Dataset(df=df, entity_col=entity_col, period_col=period_col, indicators=indicators, period_order=period_order, report=report)
# --------------------------------------------------------------------------- #
# Part B - trends
# --------------------------------------------------------------------------- #
def compute_trends(ds: Dataset, cfg: Config) -> pd.DataFrame:
    cfg.validate()
    rows = []
    for ind in ds.indicators:
        for entity in sorted(ds.df["entity"].unique()):
            series = ds.df.loc[ds.df["entity"] == entity, ["period", ind]].copy()
            series = series.sort_values("period")
            if len(series) < 2:
                continue
            for idx in range(1, len(series)):
                prev_period = series.iloc[idx - 1]["period"]
                curr_period = series.iloc[idx]["period"]
                prev_val = series.iloc[idx - 1][ind]
                curr_val = series.iloc[idx][ind]
                if pd.isna(prev_val) or pd.isna(curr_val):
                    continue
                if abs(float(prev_val)) < 1e-12:
                    continue
                change = float(curr_val) - float(prev_val)
                change_pct = (change / float(prev_val)) * 100.0
                direction = get_direction(ind, cfg)
                if direction == "lower_better":
                    good = change_pct < 0
                    status = "improved" if good else "worsened"
                else:
                    good = change_pct > 0
                    status = "improved" if good else "worsened"
                is_significant = abs(change_pct) >= float(cfg.trend_threshold)
                ratio = abs(change_pct) / float(cfg.trend_threshold)
                worse = status == "worsened"
                severity = _severity(np.array([ratio]), worse, cfg)[0]
                if change_pct < 0:
                    verb = "dropped"
                    text = f"dropped by {abs(change_pct):.1f}%"
                elif change_pct > 0:
                    verb = "rose"
                    text = f"rose by {abs(change_pct):.1f}%"
                else:
                    verb = "was flat"
                    text = "was flat"
                explanation = (
                    f"{entity}'s {pretty(ind)} {text} from {fmt(prev_val)} to {fmt(curr_val)} "
                    f"({change_pct:+.1f}%, {status})."
                )
                rows.append({
                    "entity": entity,
                    "period": curr_period,
                    "indicator": ind,
                    "value": curr_val,
                    "prev_value": prev_val,
                    "change": change,
                    "change_pct": change_pct,
                    "status": status,
                    "is_significant": is_significant,
                    "severity": severity,
                    "prev_period": prev_period,
                    "explanation": explanation,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["entity", "period", "indicator", "value", "prev_value", "change", "change_pct", "status", "is_significant", "severity", "prev_period", "explanation"])
    return out.sort_values(["entity", "indicator", "period"]).reset_index(drop=True)
# --------------------------------------------------------------------------- #
# Part C - outliers
# --------------------------------------------------------------------------- #
def compute_outliers(ds: Dataset, cfg: Config) -> pd.DataFrame:
    cfg.validate()
    method = cfg.outlier_method
    thr = cfg.outlier_thr()
    rows = []
    by_period = resolve_by_period(ds, cfg)
    grouped = ds.df.groupby("period") if by_period else [(None, ds.df)]
    if by_period:
        iterable = ds.df.groupby("period")
    else:
        iterable = [(None, ds.df)]
    for period, g in iterable:
        values_by_ind = {}
        for ind in ds.indicators:
            series = g[ind].dropna()
            if series.empty:
                continue
            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                low = q1 - thr * iqr
                high = q3 + thr * iqr
                mask = (series < low) | (series > high)
                scores = np.abs((series - series.median()) / (iqr if iqr else 1.0))
            elif method == "zscore":
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0:
                    continue
                z = (series - mean) / std
                mask = z.abs() > thr
                scores = z.abs()
            else:
                med = series.median()
                mad = (series - med).abs().median()
                if mad == 0:
                    continue
                score = 0.6745 * (series - med) / mad
                mask = score.abs() > thr
                scores = score.abs()
            flagged = series[mask]
            for row_idx, value in flagged.items():
                entity = g.loc[row_idx, "entity"]
                row_period = g.loc[row_idx, "period"]
                score = float(scores.loc[row_idx]) if row_idx in scores.index else 0.0
                severity = _severity(np.array([abs(score) / thr]), False, cfg)[0]
                rows.append({
                    "entity": entity,
                    "period": row_period if period is None else period,
                    "indicator": ind,
                    "value": value,
                    "score": score,
                    "threshold": thr,
                    "severity": severity,
                    "explanation": f"{entity}'s {pretty(ind)} is an outlier at {fmt(value)} in {period if period is not None else 'the overall dataset'}.",
                })
    if not rows:
        return pd.DataFrame(columns=["entity", "period", "indicator", "value", "score", "threshold", "severity", "explanation"])
    return pd.DataFrame(rows).sort_values(["period", "indicator", "entity"]).reset_index(drop=True)
# --------------------------------------------------------------------------- #
# Part D - threshold breaches & quality checks
# --------------------------------------------------------------------------- #
def compute_breaches(ds: Dataset, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    quality_rows = []
    for ind in ds.indicators:
        values = ds.df[ind]
        if ind in cfg.targets:
            kind, limit = cfg.targets[ind]
            if kind == "min":
                mask = values < limit
            elif kind == "max":
                mask = values > limit
            else:
                mask = pd.Series(False, index=ds.df.index)
            for idx, flag in mask.items():
                if not flag:
                    continue
                entity = ds.df.loc[idx, "entity"]
                period = ds.df.loc[idx, "period"]
                value = ds.df.loc[idx, ind]
                direction = "below" if kind == "min" else "above"
                severity = _severity(np.array([abs(float(value) - float(limit)) / max(float(cfg.trend_threshold), 1.0)]), True, cfg)[0]
                rows.append({
                    "entity": entity,
                    "period": period,
                    "indicator": ind,
                    "value": value,
                    "limit": limit,
                    "direction": direction,
                    "threshold": cfg.trend_threshold,
                    "severity": severity,
                    "breach_kind": "target",
                    "explanation": f"{entity}'s {pretty(ind)} is {direction} the target of {fmt(limit)} ({fmt(value)}).",
                })
        if is_pct_like(ind, values, cfg):
            bad = (values < 0) | (values > 100)
            for idx, flag in bad.items():
                if not flag:
                    continue
                entity = ds.df.loc[idx, "entity"]
                period = ds.df.loc[idx, "period"]
                value = ds.df.loc[idx, ind]
                severity = "High" if abs(float(value)) > 100 else "Medium"
                quality_rows.append({
                    "entity": entity,
                    "period": period,
                    "indicator": ind,
                    "value": value,
                    "n_invalid": 1,
                    "severity": severity,
                    "breach_kind": "quality",
                    "explanation": f"{entity}'s {pretty(ind)} is outside the logical 0-100% range at {fmt(value)}.",
                })
                rows.append({
                    "entity": entity,
                    "period": period,
                    "indicator": ind,
                    "value": value,
                    "limit": 100,
                    "direction": "outside",
                    "threshold": cfg.trend_threshold,
                    "severity": severity,
                    "breach_kind": "quality",
                    "explanation": f"{entity}'s {pretty(ind)} is outside the logical 0-100% range at {fmt(value)}.",
                })
    breaches = pd.DataFrame(rows)
    quality = pd.DataFrame(quality_rows)
    if breaches.empty:
        breaches = pd.DataFrame(columns=["entity", "period", "indicator", "value", "limit", "direction", "threshold", "severity", "breach_kind", "explanation"])
    if quality.empty:
        quality = pd.DataFrame(columns=["entity", "period", "indicator", "value", "n_invalid", "severity", "breach_kind", "explanation"])
    return breaches, quality
# --------------------------------------------------------------------------- #
# Part E - correlation
# --------------------------------------------------------------------------- #
def compute_correlation(ds: Dataset, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg.validate()
    corr = ds.df[ds.indicators].corr(method="pearson")
    corr = corr.fillna(0)
    valid = [c for c in corr.columns if corr[c].nunique() > 1]
    corr = corr.loc[valid, valid]
    pairs = []
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            r = float(corr.loc[a, b])
            if np.isnan(r):
                continue
            if abs(r) < cfg.corr_threshold:
                continue
            n = ds.df[[a, b]].dropna().shape[0]
            if n < 2:
                continue
            p_value = stats.pearsonr(ds.df[a].dropna(), ds.df[b].dropna())[1] if n >= 3 else 1.0
            pairs.append({
                "indicator_a": a,
                "indicator_b": b,
                "r": r,
                "p_value": p_value,
                "n": n,
            })
    pairs = pd.DataFrame(pairs)
    if pairs.empty:
        pairs = pd.DataFrame(columns=["indicator_a", "indicator_b", "r", "p_value", "n"])
    return corr, pairs
# --------------------------------------------------------------------------- #
# Part F - insight assembly
# --------------------------------------------------------------------------- #
def build_insights(ds: Dataset, cfg: Config, trends: pd.DataFrame, outliers: pd.DataFrame, breaches: pd.DataFrame, corr: pd.DataFrame, corr_pairs: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in trends[trends["is_significant"]].iterrows():
        rows.append({
            "insight_id": "",
            "type": "trend",
            "indicator": row["indicator"],
            "entity": row["entity"],
            "period": row["period"],
            "metric": row["indicator"],
            "value": row["value"],
            "prev_value": row["prev_value"],
            "change": row["change"],
            "change_pct": row["change_pct"],
            "severity": row["severity"],
            "explanation": row["explanation"],
        })
    for _, row in outliers.iterrows():
        rows.append({
            "insight_id": "",
            "type": "outlier",
            "indicator": row["indicator"],
            "entity": row["entity"],
            "period": row["period"],
            "metric": row["indicator"],
            "value": row["value"],
            "prev_value": None,
            "change": None,
            "change_pct": None,
            "severity": row["severity"],
            "explanation": row["explanation"],
        })
    for _, row in breaches.iterrows():
        if row.get("breach_kind") == "quality":
            continue
        rows.append({
            "insight_id": "",
            "type": "threshold_breach",
            "indicator": row["indicator"],
            "entity": row["entity"],
            "period": row["period"],
            "metric": row["indicator"],
            "value": row["value"],
            "prev_value": None,
            "change": None,
            "change_pct": None,
            "severity": row["severity"],
            "explanation": row["explanation"],
        })
    # Data quality signals are included in the quality frame and should also show in insights.
    for _, row in quality.iterrows():
        rows.append({
            "insight_id": "",
            "type": "threshold_breach",
            "indicator": row["indicator"],
            "entity": row["entity"],
            "period": row["period"],
            "metric": row["indicator"],
            "value": row["value"],
            "prev_value": None,
            "change": None,
            "change_pct": None,
            "severity": row["severity"],
            "explanation": row["explanation"],
        })
    # Compound/correlation insight: if a district-period has multiple adverse signals, combine them.
    if cfg.include_compound:
        for _, row in ds.df.iterrows():
            entity = row["entity"]
            period = row["period"]
            bad = []
            for ind in ds.indicators:
                value = row[ind]
                if pd.isna(value):
                    continue
                direction = get_direction(ind, cfg)
                if direction == "lower_better":
                    bad_flag = value > 0 and value >= (ds.df[ind].quantile(0.75) if ds.df[ind].notna().any() else 0)
                else:
                    bad_flag = value < (ds.df[ind].quantile(0.25) if ds.df[ind].notna().any() else 0)
                if bad_flag and ind not in ("entity", "period"):
                    bad.append(ind)
            if len(bad) >= 2:
                pair = bad[:2]
                rows.append({
                    "insight_id": "",
                    "type": "correlation",
                    "indicator": f"{pair[0]}:{pair[1]}",
                    "entity": entity,
                    "period": period,
                    "metric": f"{pair[0]}:{pair[1]}",
                    "value": None,
                    "prev_value": None,
                    "change": None,
                    "change_pct": None,
                    "severity": "Medium",
                    "explanation": f"{entity} shows a compound risk pattern across {pretty(pair[0])} and {pretty(pair[1])}.",
                })
    if rows:
        out = pd.DataFrame(rows, columns=INSIGHT_COLUMNS)
        out["insight_id"] = [f"INS-{i + 1:04d}" for i in range(len(out))]
        return out
    return pd.DataFrame(columns=INSIGHT_COLUMNS)
# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_analysis(ds: Dataset, cfg: Config, filters: Optional[Filters] = None) -> Analysis:
    cfg.validate()
    if filters is not None:
        if filters.entities is not None:
            ds.df = ds.df[ds.df["entity"].isin(filters.entities)].copy()
        if filters.periods is not None:
            ds.df = ds.df[ds.df["period"].isin(filters.periods)].copy()
        if filters.indicators is not None:
            ds.df = ds.df[["entity", "period", *[c for c in filters.indicators if c in ds.indicators]]].copy()
            ds.indicators = [c for c in filters.indicators if c in ds.indicators]
    trends = compute_trends(ds, cfg)
    outliers = compute_outliers(ds, cfg)
    breaches, quality = compute_breaches(ds, cfg)
    corr, corr_pairs = compute_correlation(ds, cfg)
    insights = build_insights(ds, cfg, trends, outliers, breaches, corr, corr_pairs, quality)
    notes = list(ds.report.get("warnings", []))
    if ds.df["entity"].nunique() < 10:
        notes.append("Fewer than 10 entities in the selected data; outlier/correlation results should be treated as indicative.")
    if len(ds.df["period"].unique()) < 2:
        notes.append("Only one period is available; trend analysis is skipped.")
    return Analysis(
        trends=trends,
        outliers=outliers,
        breaches=breaches,
        corr=corr,
        corr_pairs=corr_pairs,
        quality=quality,
        insights=insights,
        notes=notes,
    )
# Backward-compatible alias for the legacy sample project
# This is used by tests and by the Streamlit app.
def compute_compound_insights(ds: Dataset, cfg: Config):
    return build_insights(ds, cfg, compute_trends(ds, cfg), compute_outliers(ds, cfg), compute_breaches(ds, cfg)[0], *compute_correlation(ds, cfg), compute_breaches(ds, cfg)[1])
