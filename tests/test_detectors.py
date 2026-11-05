import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detectors import (
    d1_missing,
    d2_pattern,
    d3_robust_z,
    d4_multivariate,
    d5_fd,
    d6_typo,
    build_candidates
)

def make_sample_dirty():
    return pd.DataFrame({
        "city": ["birmingham", "birmingham", "birmingham", "birmingxam", "sheffield", "sheffield", "sheffield", ""],
        "state": ["al", "al", "al", "al", "al", "al", "al", "al"],
        "zip": ["35233", "35233", "35233", "35233", "35660", "35660", "35660", "99999"],
        "val": [10.0, 10.5, 9.8, 10.2, 11.0, 10.1, 10.4, 500.0]
    })

def test_detectors_only_accept_dirty_df():
    df = make_sample_dirty()
    for det in (d1_missing.detect, d2_pattern.detect, d3_robust_z.detect, d4_multivariate.detect, d5_fd.detect, d6_typo.detect):
        res = det(df)
        assert isinstance(res, pd.DataFrame)
        assert set(res.columns) == {"row", "column", "detector", "score"}

def test_d1_detects_empty():
    df = make_sample_dirty()
    res = d1_missing.detect(df)
    assert not res.empty
    hit = res[(res["row"] == 7) & (res["column"] == "city")]
    assert len(hit) == 1
    assert hit.iloc[0]["score"] == 1.0

def test_d3_detects_outlier():
    df = make_sample_dirty()
    # add enough values so n >= 10
    more_rows = pd.DataFrame({
        "city": ["birmingham"] * 10,
        "state": ["al"] * 10,
        "zip": ["35233"] * 10,
        "val": [10.0] * 10
    })
    df_large = pd.concat([df, more_rows], ignore_index=True)
    res = d3_robust_z.detect(df_large)
    hit = res[(res["row"] == 7) & (res["column"] == "val")]
    assert len(hit) == 1

def test_d6_detects_typo():
    # birmingxam (1 time) vs frequent birmingham (20 times)
    df = pd.DataFrame({
        "city": ["birmingham"] * 20 + ["birmingxam"] + ["sheffield"] * 10
    })
    res = d6_typo.detect(df)
    hit = res[res["column"] == "city"]
    assert not hit.empty
    assert any(h["score"] >= 0.75 for _, h in hit.iterrows())

def test_build_candidates():
    df = make_sample_dirty()
    cands = build_candidates(df, table="toy", split="val")
    assert "is_candidate" in cands.columns
    assert "d1" in cands.columns
    assert len(cands) == len(df) * len(df.columns)
    # Row 7, city is missing -> must be candidate
    missing_cell = cands[(cands["row"] == 7) & (cands["column"] == "city")]
    assert missing_cell.iloc[0]["is_candidate"]
    assert missing_cell.iloc[0]["d1"] == 1.0


def test_detectors_never_read_clean_files(monkeypatch):
    def forbidden_open(*args, **kwargs):
        raise AssertionError("Detectors must never access the filesystem or clean files!")
    
    df = make_sample_dirty()
    with monkeypatch.context() as m:
        m.setattr("builtins.open", forbidden_open)
        for det in (d1_missing.detect, d2_pattern.detect, d3_robust_z.detect, d4_multivariate.detect, d5_fd.detect, d6_typo.detect):
            res = det(df)
            assert isinstance(res, pd.DataFrame)
