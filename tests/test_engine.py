"""
Unit tests for engine.py.   Run from the project root:  pytest -q

The PDF's 12-row sample is embedded below because it is the only data with
known planted patterns (Ahmedabad ANC drop, Mehsana outlier, Mehsana anomaly).
"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine as E  # noqa: E402

SAMPLE = """month,district,anc_coverage,institutional_delivery,immunization,high_risk_cases
2026-07,Ahmedabad,85,91,93,10
2026-08,Ahmedabad,69,90,92,13
2026-07,Surat,81,87,91,13
2026-08,Surat,83,89,92,12
2026-07,Vadodara,90,93,96,7
2026-08,Vadodara,91,94,97,6
2026-07,Rajkot,79,85,89,15
2026-08,Rajkot,80,86,90,14
2026-07,Mehsana,84,89,92,11
2026-08,Mehsana,42,88,91,28
2026-07,Bhavnagar,77,82,87,17
2026-08,Bhavnagar,79,84,89,15
"""

REQUIRED = ["insight_id", "type", "indicator", "entity", "period", "metric",
            "value", "prev_value", "change", "change_pct", "severity", "explanation"]


def run(csv_text: str, **cfg):
    ds = E.load_data(io.StringIO(csv_text))
    return ds, E.run_analysis(ds, E.Config(**cfg))


@pytest.fixture(scope="module")
def sample():
    return run(SAMPLE)


def test_ahmedabad_anc_trend_is_minus_18_8_and_high(sample):
    ins = sample[1].insights
    row = ins[(ins.type == "trend") & (ins.entity == "Ahmedabad")
              & (ins.indicator == "anc_coverage")].iloc[0]
    assert row.change_pct == pytest.approx(-18.82, abs=0.05)
    assert row.severity == "High"
    assert "dropped by 18.8%" in row.explanation


def test_mehsana_anc_is_an_outlier(sample):
    ins = sample[1].insights
    out = ins[(ins.type == "outlier") & (ins.entity == "Mehsana")
              & (ins.indicator == "anc_coverage")]
    assert len(out) == 1 and out.iloc[0].value == 42


def test_mehsana_compound_insight_links_anc_and_high_risk(sample):
    ins = sample[1].insights
    comp = ins[(ins.type == "correlation") & (ins.entity == "Mehsana")]
    assert "anc_coverage" in comp.iloc[0].indicator
    assert "high_risk_cases" in comp.iloc[0].indicator


def test_insights_have_all_required_fields_and_sequential_ids(sample):
    ins = sample[1].insights
    assert list(ins.columns) == REQUIRED
    assert ins.insight_id.tolist() == [f"INS-{i + 1:04d}" for i in range(len(ins))]
    assert set(ins.severity) <= {"Low", "Medium", "High"}
    assert set(ins.type) <= {"trend", "outlier", "correlation", "threshold_breach"}


def test_no_hardcoded_narratives():
    """Rename every district: no old name may survive in any explanation."""
    renamed = SAMPLE
    for old, new in [("Ahmedabad", "Alpha"), ("Surat", "Beta"), ("Vadodara", "Gamma"),
                     ("Rajkot", "Delta"), ("Mehsana", "Epsilon"), ("Bhavnagar", "Zeta")]:
        renamed = renamed.replace(old, new)
    _, an = run(renamed)
    text = " ".join(an.insights.explanation)
    assert "Alpha" in text and "Epsilon" in text
    for old in ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Mehsana", "Bhavnagar"]:
        assert old not in text


def test_higher_trend_threshold_gives_fewer_significant_trends():
    ds = E.load_data(io.StringIO(SAMPLE))
    low = E.compute_trends(ds, E.Config(trend_threshold=5)).is_significant.sum()
    high = E.compute_trends(ds, E.Config(trend_threshold=40)).is_significant.sum()
    assert high < low


def test_zero_previous_value_does_not_crash_or_produce_inf():
    csv = ("month,district,x\n2026-01,A,0\n2026-02,A,5\n2026-03,A,6\n"
           "2026-01,B,2\n2026-02,B,3\n2026-03,B,4\n")
    ds, an = run(csv)
    t = an.trends
    assert not (t.change_pct.abs() == float("inf")).any()
    assert not ((t.entity == "A") & (t.period == "2026-02")).any()   # undefined, skipped


def test_single_period_district_is_skipped_with_warning():
    csv = SAMPLE + "2026-08,Lonely,70,80,90,5\n"
    ds, an = run(csv)
    assert any("only one" in w for w in ds.report["warnings"])
    assert "Lonely" not in set(an.trends.entity)


def test_direction_override_changes_status():
    ds = E.load_data(io.StringIO(SAMPLE))
    base = E.compute_trends(ds, E.Config())
    flipped = E.compute_trends(ds, E.Config(directions={"anc_coverage": "lower_better"}))
    pick = lambda t: t[(t.entity == "Ahmedabad") & (t.indicator == "anc_coverage")].status.iloc[0]
    assert pick(base) == "worsened" and pick(flipped) == "improved"


def test_high_risk_rise_is_adverse_by_default():
    ds = E.load_data(io.StringIO(SAMPLE))
    t = E.compute_trends(ds, E.Config())
    row = t[(t.entity == "Mehsana") & (t.indicator == "high_risk_cases")].iloc[0]
    assert row.status == "worsened"


def test_all_three_outlier_methods_run():
    ds = E.load_data(io.StringIO(SAMPLE))
    for m in E.OUTLIER_METHODS:
        E.run_analysis(ds, E.Config(outlier_method=m))


def test_correlation_matrix_matches_pandas(sample):
    ds, an = sample
    expected = ds.df[ds.indicators].corr()
    pd.testing.assert_frame_equal(an.corr, expected)


def test_constant_column_does_not_break_correlation():
    csv = ("month,district,a,b\n2026-01,A,1,5\n2026-02,A,2,5\n2026-01,B,3,5\n2026-02,B,4,5\n")
    _, an = run(csv)           # column b is constant -> r undefined -> pair skipped
    assert an.corr_pairs.empty


def test_missing_required_columns_raise_clear_error():
    with pytest.raises(ValueError, match="No time column"):
        E.load_data(io.StringIO("district,x\nA,1\n"))
    with pytest.raises(ValueError, match="No entity column"):
        E.load_data(io.StringIO("month,x\n2026-01,1\n"))


def test_percentage_above_100_is_flagged_as_data_quality():
    csv = ("fiscal_year,district,anc_coverage\n2020-21,A,105\n2021-22,A,99\n"
           "2020-21,B,90\n2021-22,B,95\n")
    _, an = run(csv)
    assert len(an.quality) == 1 and an.quality.iloc[0].n_invalid == 1
    assert (an.insights.type == "threshold_breach").any()


def test_invalid_config_is_rejected():
    ds = E.load_data(io.StringIO(SAMPLE))
    with pytest.raises(ValueError):
        E.run_analysis(ds, E.Config(medium_mult=3.0, high_mult=2.0))
