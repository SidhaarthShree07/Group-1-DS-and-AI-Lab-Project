#!/usr/bin/env python3
"""Detector D7: numeric columns written in more than one format.

Author: Sidhaarth Shree

D2 flags rare patterns, so it misses an error when the wrong format is the common one in its column. In Beers,
"12.0 oz" is written for "12", "0.05%" for "0.05" and "N/A" for an empty value; 118 of the 448 errors never became
candidates for this reason. This detector looks only at the dirty table and uses no column names.

For each column:
  - every value is classed as plain number, decorated number ("12.0 oz", "0.05%", "$4"), placeholder ("n/a",
    "null", "-") or other text;
  - the column counts as numeric when at least MIN_NUMERIC_SHARE of its values are numbers (placeholders count
    neither way);
  - mixed formats: plain and decorated/placeholder values both make up at least MIN_FORMAT_SHARE of the column;
  - mixed units: every value is decorated, but a second way of writing the unit ("oz." next to "oz", "ounce")
    makes up at least MIN_SECOND_UNIT_SHARE of the values. A rare second form ("1 patient" among "N patients")
    does not count.
In a numeric column with mixed formats or mixed units, every decorated or placeholder cell is flagged. The model
and the gate still decide each flagged cell.

This replaced the stage 0 rules (the first D7), which added almost no recall in E6.
"""
from __future__ import annotations

import re

import pandas as pd

COLUMNS = ["row", "column", "detector", "score"]
MIN_NUMERIC_SHARE = 0.80
MIN_FORMAT_SHARE = 0.02
MIN_SECOND_UNIT_SHARE = 0.10

PLAIN_NUMBER = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)$")
DECORATED_NUMBER = re.compile(
    r"^[^\d\-+.]{0,3}\s*[-+]?(\d+(\.\d*)?|\.\d+)\s*[%a-zA-Z.$€£ ]{1,20}$|^[$€£]\s*[-+]?\d+(\.\d+)?$")
ANY_NUMBER = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)")
PLACEHOLDERS = {"n/a", "na", "null", "none", "nan", "-", "--", "?", "not available", "unknown", "missing"}


def value_format(value: str) -> str:
    """'empty', 'plain', 'placeholder', 'decorated' or 'other'."""
    value = value.strip()
    if not value:
        return "empty"
    if PLAIN_NUMBER.match(value):
        return "plain"
    if value.lower() in PLACEHOLDERS:
        return "placeholder"
    if DECORATED_NUMBER.match(value):
        return "decorated"
    return "other"


def unit_text(value: str) -> str:
    """The text around the number, lower case and without spaces: '12.0 OZ.' -> 'oz.'."""
    return ANY_NUMBER.sub("", value.strip().lower()).replace(" ", "")


def column_needs_flags(values: list, formats: list) -> bool:
    filled = [f for f in formats if f != "empty"]
    if not filled:
        return False
    n = len(filled)
    n_plain = filled.count("plain")
    n_decorated = filled.count("decorated")
    n_placeholder = filled.count("placeholder")
    if n_placeholder == n or (n_plain + n_decorated) / (n - n_placeholder) < MIN_NUMERIC_SHARE:
        return False

    mixed_formats = n_plain / n >= MIN_FORMAT_SHARE and (n_decorated + n_placeholder) / n >= MIN_FORMAT_SHARE
    units = pd.Series([unit_text(v) for v, f in zip(values, formats) if f == "decorated"]).value_counts()
    mixed_units = len(units) >= 2 and units.iloc[1] / units.sum() >= MIN_SECOND_UNIT_SHARE
    return mixed_formats or mixed_units


def detect(dirty_df: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for col in dirty_df.columns:
        values = [str(v) for v in dirty_df[col]]
        formats = [value_format(v) for v in values]
        if column_needs_flags(values, formats):
            flags += [(row, col) for row, f in enumerate(formats) if f in ("decorated", "placeholder")]
    out = pd.DataFrame(flags, columns=["row", "column"])
    out["detector"] = "d7"
    out["score"] = 1.0
    return out[COLUMNS]
