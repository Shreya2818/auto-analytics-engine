# Auto-Analytics Engine: Automated Insight Generation

A general-purpose engine that ingests a district-level healthcare CSV and automatically
finds **trends, outliers and correlations**, then writes **plain-English insights** with a
severity (Low / Medium / High). A Streamlit UI lets you filter the data and tune every
threshold live.

It is rule-based statistics (no ML model, no database, no LLM). Every number in an insight
sentence is read from the data, so each insight can be traced and audited.

## Project structure

```
auto-insight-engine/
├── data/districts.csv        # Gujarat HMIS data, 33 districts x 5 fiscal years (165 rows)
├── dataset/                  # raw HMIS download (long format, raw counts)
├── prepare_data.py           # raw HMIS -> data/districts.csv
├── engine.py                 # all analytics logic (no UI code)
├── app.py                    # Streamlit UI
├── tests/test_engine.py      # pytest suite
├── outputs/                  # insights.csv / insights.json / correlation_matrix.csv (from the app)
├── screenshots/
├── requirements.txt
└── README.md
```

## How to run

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pytest -q`.

To rebuild the dataset from the raw HMIS download:

```bash
python prepare_data.py --raw dataset/<raw-hmis-file>.csv --out data/districts.csv
```

In the app, click **Save outputs to ./outputs/** to write `insights.csv`, `insights.json`
and `correlation_matrix.csv`. You can also upload any CSV with an entity column
(`district`, `entity`, `region`...), a time column (`month`, `period`, `fiscal_year`, `year`...)
and one or more numeric indicators. Nothing in the engine is tied to specific names.

## Data

**Source:** Health Management Information System (HMIS), India, "District and Year-wise data
related to all Health Indicators for Gujarat", via Dataful (https://dataful.in/datasets/14270/14270),
GODL-style open government data. Raw counts, yearly, 33 districts, 2008-09 to 2021-22.

**Window used:** 2017-18 to 2021-22. Before 2017-18 HMIS recorded "3 ANC check-ups"; from
2017-18 it records "4 or more". Mixing them would create a fake trend at the boundary.

| Column | Definition |
|---|---|
| `anc_coverage` | women with 4+ ANC check-ups / women registered for ANC x 100 |
| `institutional_delivery` | institutional deliveries / (institutional + home deliveries) x 100 |
| `immunization` | Pentavalent-3 doses / live births (male + female) x 100 |
| `high_risk_cases` | hypertension detected in pregnancy + severe anaemia (Hb<7) treated + eclampsia managed (raw count) |

Pentavalent-3 is used instead of DPT3 because Gujarat reports the Pentavalent vaccine.

## What the engine does

| Part | Function | Method |
|---|---|---|
| A. Load and validate | `load_data` | detects entity/time/indicator columns, drops duplicates, reports `head()`, `info()`, missing values and warnings |
| B. Trends | `compute_trends` | `pct_change = (current - previous) / previous x 100` per (district, indicator); significant if `abs(pct_change) >= threshold` (default 10%) |
| C. Outliers | `compute_outliers` | IQR rule, Z-score or modified Z-score (median/MAD), selectable; compared with peers in the same period when there are 10+ districts |
| D. Correlation | `compute_correlation` | Pearson matrix (`DataFrame.corr()`), pairs flagged at `abs(r) >= 0.70`, with n and p-value |
| E. Insights | `build_insights` | template sentences, numbers from data; fields `insight_id, type, indicator, entity, period, metric, value, prev_value, change, change_pct, severity, explanation` |
| F. UI | `app.py` | filters, sliders, insight table, severity bar charts, correlation heatmap, per-district line chart |

Insight types: `trend`, `outlier`, `correlation`, `threshold_breach`.

* **Threshold breach** is user-defined: set a minimum or maximum target for an indicator in the
  sidebar. Percentage columns that leave the logical 0-100% range are also reported as
  threshold breaches (a data-quality signal).
* **Compound insight** (type `correlation`, indicator `a:b`): the same district has two or more
  adverse flags in the same period. This is how a collapse in one indicator alongside a spike in
  another becomes one actionable finding.

### Severity formula (no magic numbers)

```
ratio    = signal strength / threshold
           trend:   abs(pct_change) / trend threshold
           outlier: score / outlier threshold
           breach:  gap to target (%) / trend threshold
level    = Low     if ratio <  medium_mult        (default 1.5)
           Medium  if medium_mult <= ratio < high_mult
           High    if ratio >= high_mult          (default 2.0)
severity = level + 1 step if the change is in the BAD direction (capped at High)
```

Both multipliers are sliders. Correlation severity grows with how far `abs(r)` lies between the
threshold and 1, and is capped at Low when n is small or p >= 0.05.

**Direction matters:** a drop in ANC coverage is bad, a rise in high-risk cases is bad. Direction
is guessed from the column name (words such as risk, death, cases) and can be overridden per
indicator in the sidebar.

## Results on the Gujarat data (default settings)

268 insights: 188 trend, 54 outlier, 24 correlation (all compound, co-occurring flags) and
2 threshold breach (data quality); 182 High, 55 Medium, 31 Low.

* The default 10% trend threshold is very sensitive for **yearly** data. Raising it to 25% gives 149
  insights (82 High). Use the slider to match the decision you need to support (alert fatigue).
* No pair of indicators reaches `abs(r) >= 0.70` (all are between -0.25 and +0.13), so no global
  correlation insight is raised. The heatmap shows this honestly.

## Assumptions

* Direction of each indicator (higher or lower is better) is inferred from its name unless overridden.
* Threshold-breach targets are not in the data, so they are user-defined.
* "Previous period" means the previous fiscal year in this dataset (previous month for monthly files).
* Public + private facilities combined ("Total") are used, to avoid double counting.

## Limitations

* **Few periods per district:** only 5 yearly points, so trends are one-step comparisons and not forecasts.
* **Correlations are indicative, not conclusive.** Rows from the same district are not independent,
  and the app displays n beside every correlation. The PDF's 12-row sample is far weaker (n = 12).
* **HMIS data quality:** `immunization` exceeds 100% in 87 of 165 rows (children vaccinated vs births
  registered come from different counts) and `anc_coverage` in 2. These values are kept, not capped,
  and flagged. Large year-on-year jumps (for example high-risk cases in some districts in 2018-19)
  may reflect reporting changes, not real events.
* `high_risk_cases` is a raw count, so large districts naturally have larger numbers.
* No causality is claimed anywhere.

## Future work

Forecasting and seasonality for monthly data, population-adjusted rates, DHIS2/HMIS live
connection, SQLite/PostgreSQL storage for many years, alert e-mails.
