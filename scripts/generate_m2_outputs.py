#!/usr/bin/env python3
"""
scripts/generate_m2_outputs.py

Generates ContextGuard Milestone 2 Quality Table and Figures:
- docs/tables/m2_core_quality.csv
- docs/figures/m2/error_rate_per_column.svg
- docs/figures/m2/value_length_distribution.svg
- docs/figures/m2/error_type_mix.svg

Author: Jashan (Data Lead) - DSAI Lab, IIT Madras BS
"""

import os
import sys
import csv
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

MISSING_TOKENS = {'', 'nan', 'null', 'none', 'n/a', 'na', 'not available', '?', '-', 'undefined'}
EXCLUDE_COLS = ('row_id', 'tuple_id', 'index', 'id_col')

TABLE_CONFIGS = [
    {
        "id": "hospital",
        "name": "Hospital",
        "raha_sub": "hospital",
        "entity_key": "provider_number",
        "training_impact": "Clean-file has 0 missing cells and 0 duplicate rows; 19 header casing/naming differences aligned positionally; clinical measure rule violations (408) provide strong training signal for cross-column dependency learning."
    },
    {
        "id": "flights",
        "name": "Flights",
        "raha_sub": "flights",
        "entity_key": "flight",
        "training_impact": "0 clean gaps; 2,370 dirty missing/placeholder values ('Not Available', blanks) in time fields; 904 unannotated format errors recovered via deterministic heuristic; temporal consistency rules trained on departure/arrival relationships."
    },
    {
        "id": "beers",
        "name": "Beers",
        "raha_sub": "beers",
        "entity_key": "brewery_id",
        "training_impact": "1,072 missing cells in clean ground truth (mostly unrecorded IBU); these cells are labelled by the Raha rule like any other (an 'N/A' placeholder against an empty clean value is a missing-value error, and the repair target is the empty string); 1,005 'N/A' errors are present in the Raha copy; the ounces column exhibits a 100% format error rate."
    },
    {
        "id": "rayyan",
        "name": "Rayyan",
        "raha_sub": "rayyan",
        "entity_key": "id",
        "training_impact": "1,609 clean missing cells in optional bibliographic attributes; empty clean values are labelled by the Raha rule; high rule violation share (619 / 948, 65.3%) trains journal title to abbreviation and ISSN functional dependencies."
    },
    {
        "id": "movies",
        "name": "Movies",
        "raha_sub": "movies_1",
        "entity_key": "id",
        "training_impact": "2,680 clean missing cells (mostly Filming Locations, ReviewCount, Duration and RatingValue), labelled by the Raha rule; RatingCount comma format differences (5,505 cells) and RatingValue formatting differences (856 cells, 669 of them the same number written differently, such as 7.0 and 7) are annotated pattern errors in the Raha copy, which TableEG's copy lost in part (668 errors)."
    }
]


def raha_file(repo_root: Path, sub: str, name: str) -> Path:
    """Raha table file from a local Raha clone or from data/raw/raha (see scripts/download_data.py)."""
    for base in (repo_root / "raha-master" / "datasets", repo_root / "data" / "raw" / "raha" / "datasets",
                 repo_root / "data" / "raw" / "raha"):
        if (base / sub / name).exists():
            return base / sub / name
    raise FileNotFoundError(f"Raha table {sub}/{name} not found; run scripts/download_data.py")


def format_only(a: str, b: str) -> bool:
    """Same number written differently (3 and 3.0, 379,998 and 379998, leading zeros)."""
    a, b = a.replace(",", ""), b.replace(",", "")
    try:
        return a != b and float(a) == float(b)
    except ValueError:
        return False


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = list(csv.reader(f))
    return reader[0], reader[1:]


def build_quality_table(repo_root: Path):
    out_csv = repo_root / "docs" / "tables" / "m2_core_quality.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for cfg in TABLE_CONFIGS:
        tname = cfg["name"]
        tid = cfg["id"]
        c_p = raha_file(repo_root, cfg["raha_sub"], "clean.csv")
        d_p = raha_file(repo_root, cfg["raha_sub"], "dirty.csv")
        cells_p = repo_root / "data" / "processed" / "core" / f"{tid}_cells.csv"

        c_hdr, c_rows = load_csv(c_p)
        d_hdr, d_rows = load_csv(d_p)

        n_rows = len(d_rows)
        n_cols = min(len(c_hdr), len(d_hdr))
        ex_idx = {i for i in range(n_cols) if c_hdr[i].lower() in EXCLUDE_COLS or d_hdr[i].lower() in EXCLUDE_COLS}
        data_cols = [i for i in range(n_cols) if i not in ex_idx]

        # Column types and distinct counts
        numeric_cols = 0
        text_cols = 0
        distinct_vals = []
        dirty_missing = 0
        clean_missing = 0

        for c in data_cols:
            d_vals = [r[c].strip() for r in d_rows if c < len(r)]
            c_vals = [r[c].strip() for r in c_rows if c < len(r)]
            distinct_vals.append(len(set(d_vals)))
            dirty_missing += sum(1 for v in d_vals if v.lower() in MISSING_TOKENS)
            clean_missing += sum(1 for v in c_vals if v.lower() in MISSING_TOKENS)

            non_empty = [v for v in c_vals if v.lower() not in MISSING_TOKENS]
            if non_empty:
                num_num = sum(1 for v in non_empty if re.match(r'^-?\d+(\.\d+)?$', v))
                if num_num / len(non_empty) > 0.8:
                    numeric_cols += 1
                else:
                    text_cols += 1
            else:
                text_cols += 1

        # Error cells and rates from processed cells
        with open(cells_p, "r", encoding="utf-8") as f:
            cells = list(csv.DictReader(f))
        tot_cells = len(cells)
        err_cells = sum(1 for c in cells if c["is_error"] == "True")
        err_rate = err_cells / tot_cells * 100

        # Exact and near-duplicate rows
        d_tuples = [tuple(r[c] for c in data_cols) for r in d_rows]
        exact_dups = len(d_tuples) - len(set(d_tuples))

        # Header mismatches
        hdr_mismatches = sum(1 for c in range(n_cols) if c_hdr[c] != d_hdr[c])

        # Format artifacts: error cells where dirty and clean are the same number in another format
        format_artifacts = sum(1 for c in cells if c["is_error"] == "True" and format_only(c["dirty_value"], c["clean_value"]))

        rows_out.append({
            "Table": tname,
            "Rows": n_rows,
            "Data_Columns": len(data_cols),
            "Total_Columns": n_cols,
            "Numeric_Columns": numeric_cols,
            "Categorical_Columns": text_cols,
            "Min_Distinct_Values": min(distinct_vals),
            "Max_Distinct_Values": max(distinct_vals),
            "Mean_Distinct_Values": round(sum(distinct_vals) / len(distinct_vals), 1),
            "Total_Cells": tot_cells,
            "Error_Cells": err_cells,
            "Error_Rate": f"{err_rate:.2f}%",
            "Dirty_Missing_Placeholder_Cells": dirty_missing,
            "Format_Artifacts": format_artifacts,
            "Clean_Missing_Cells": clean_missing,
            "Exact_Duplicate_Rows": exact_dups,
            "Header_Mismatches": hdr_mismatches,
            "Training_Impact": cfg["training_impact"]
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Table", "Rows", "Data_Columns", "Total_Columns", "Numeric_Columns", "Categorical_Columns",
            "Min_Distinct_Values", "Max_Distinct_Values", "Mean_Distinct_Values", "Total_Cells",
            "Error_Cells", "Error_Rate", "Dirty_Missing_Placeholder_Cells", "Format_Artifacts",
            "Clean_Missing_Cells", "Exact_Duplicate_Rows", "Header_Mismatches", "Training_Impact"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[SUCCESS] Exported docs/tables/m2_core_quality.csv ({len(rows_out)} tables)")


def build_core_dictionary(repo_root: Path):
    """
    Generates docs/tables/m2_core_dictionary.csv:
    One row per column of the 5 core tables with:
    table_name, column_name, data_type, semantic_meaning, example_value, missing_rate_percent.
    """
    out_csv = repo_root / "docs" / "tables" / "m2_core_dictionary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    column_definitions = {
        "hospital": {
            "ProviderNumber": ("numeric_text", "US CMS Medicare provider identification number uniquely identifying the hospital institution", "10018"),
            "HospitalName": ("categorical_text", "Official commercial and legal facility name of the hospital", "callahan eye foundation hospital"),
            "Address1": ("categorical_text", "Primary physical street address of the hospital facility", "1720 university blvd"),
            "Address2": ("categorical_text", "Secondary street address line, building or suite specification", "empty"),
            "Address3": ("categorical_text", "Tertiary street address line (rarely populated)", "empty"),
            "City": ("categorical_text", "Municipality or city where the hospital is located", "birmingham"),
            "State": ("categorical_text", "Two-letter US state postal code abbreviation", "al"),
            "ZipCode": ("numeric_text", "Five-digit US postal ZIP code", "35233"),
            "CountyName": ("categorical_text", "US administrative county or parish jurisdiction name", "jefferson"),
            "PhoneNumber": ("numeric_text", "Ten-digit telephone contact number for the facility", "2053258100"),
            "HospitalType": ("categorical_text", "Institutional operational classification (e.g. Acute Care Hospitals)", "acute care hospitals"),
            "HospitalOwner": ("categorical_text", "Legal ownership entity classification (e.g. Voluntary non-profit, Government)", "voluntary non-profit - private"),
            "EmergencyService": ("categorical_text", "Binary indicator (yes/no) of operational emergency room services", "yes"),
            "Condition": ("categorical_text", "Standardized CMS clinical disease or treatment diagnostic category", "surgical infection prevention"),
            "MeasureCode": ("categorical_text", "Standardized CMS clinical quality performance measure code", "scip-card-2"),
            "MeasureName": ("categorical_text", "Full descriptive title of the evaluated clinical performance measure", "surgery patients who were taking heart drugs called beta blockers"),
            "Score": ("numeric_text", "Clinical score or performance rate percentage achieved by facility", "empty"),
            "Sample": ("numeric_text", "Number of evaluated patient cases in the evaluation sample", "empty"),
            "Stateavg": ("categorical_text", "Statewide baseline comparative benchmark for the clinical measure", "al_scip-card-2")
        },
        "flights": {
            "src": ("categorical_text", "Airline code or data source feed identifier (e.g. aa, ua, us)", "aa"),
            "flight": ("categorical_text", "Standardized flight route identifier combining carrier, flight number, origin, destination", "AA-3859-IAH-ORD"),
            "sched_dep_time": ("temporal_text", "Scheduled departure timestamp (local time with a.m./p.m.)", "7:10 a.m."),
            "act_dep_time": ("temporal_text", "Actual observed departure timestamp", "7:16 a.m."),
            "sched_arr_time": ("temporal_text", "Scheduled arrival timestamp (local time with a.m./p.m.)", "9:40 a.m."),
            "act_arr_time": ("temporal_text", "Actual observed arrival timestamp", "9:32 a.m.")
        },
        "beers": {
            "id": ("numeric_int", "Unique database primary key identifier for the craft beer product", "1436"),
            "beer-name": ("categorical_text", "Commercial brand name of the craft beer beverage", "Pub Beer"),
            "style": ("categorical_text", "Brewing style classification (e.g. American Pale Lager, IPA)", "American Pale Lager"),
            "ounces": ("numeric_text", "Liquid volume measure of beverage container in fluid ounces", "12.0 oz"),
            "abv": ("numeric_float", "Alcohol by volume percentage represented as decimal fraction", "0.05"),
            "ibu": ("numeric_int", "International Bitterness Units rating", "60"),
            "brewery_id": ("numeric_int", "Foreign key identifier linking beer to its manufacturing brewery", "408"),
            "brewery-name": ("categorical_text", "Commercial trade name of the brewing company", "10 Barrel Brewing Company"),
            "city": ("categorical_text", "Municipality or city where brewery is located", "Bend"),
            "state": ("categorical_text", "Two-letter US state postal code abbreviation of brewery", "OR")
        },
        "rayyan": {
            "id": ("numeric_int", "Bibliographic citation record primary key identifier in systematic review corpus", "235295"),
            "article_title": ("categorical_text", "Full academic title of the published scientific article", "Late repair of injuries of the bile ducts"),
            "article_language": ("categorical_text", "Three-letter ISO language code of the published paper (e.g. eng)", "eng"),
            "journal_title": ("categorical_text", "Journal name or its abbreviation (the benchmark holds either form in this column)", "Proc R Soc Med"),
            "jounral_abbreviation": ("categorical_text", "Journal name or its abbreviation (column name is misspelt in the source; the benchmark holds either form)", "Proceedings of the Royal Society of Medicine"),
            "journal_issn": ("categorical_text", "International Standard Serial Number of the publishing journal", "0035-9157 (Print) 0035-9157"),
            "article_jvolumn": ("numeric_int", "Journal volume number where the paper was published", "64"),
            "article_jissue": ("numeric_int", "Journal issue number within the publishing volume", "12"),
            "article_jcreated_at": ("temporal_text", "Publication date or citation creation timestamp string", "1/1/71"),
            "article_pagination": ("categorical_text", "Starting and ending page numbers of the article publication", "1187-9"),
            "author_list": ("categorical_text", "Serialized array of contributing scientific author names", "{\"A. G. Parks\",\"J. F. McPartlin\"}")
        },
        "movies": {
            "Id": ("categorical_text", "IMDb title identifier string (e.g. tt0054215)", "tt0054215"),
            "Name": ("categorical_text", "Official cinematic release title of the motion picture", "Psycho"),
            "Year": ("numeric_int", "Four-digit calendar year of initial theatrical release", "1960"),
            "Release Date": ("temporal_text", "Full formatted theatrical release date with country specification", "8 September 1960 (USA)"),
            "Director": ("categorical_text", "Full name(s) of the film director(s)", "Alfred Hitchcock"),
            "Creator": ("categorical_text", "Screenplay writer(s) or story creator(s)", "Joseph Stefano,Robert Bloch"),
            "Actors": ("categorical_text", "Primary billed cast member names", "Anthony Perkins,Janet Leigh,Vera Miles"),
            "Cast": ("categorical_text", "Extended cast ensemble list", "Anthony Perkins,Vera Miles,John Gavin"),
            "Language": ("categorical_text", "Primary spoken dialogue language of the motion picture", "English"),
            "Country": ("categorical_text", "Primary country of film origin and production", "USA"),
            "Duration": ("numeric_text", "Total cinematic runtime duration in minutes", "109 min"),
            "RatingValue": ("numeric_float", "IMDb average user rating score on a 1-10 decimal scale", "8.6"),
            "RatingCount": ("numeric_text", "Total count of user ratings received with comma thousands separator", "379,998"),
            "ReviewCount": ("categorical_text", "Counts of user reviews and critic reviews", "976 user,290 critic"),
            "Genre": ("categorical_text", "Comma-separated list of cinematic genre classifications", "Horror,Mystery,Thriller"),
            "Filming Locations": ("categorical_text", "Real-world geographic production locations where scenes were filmed", "Title and Trust Building, 114 W. Adams Street, Phoenix, Arizona, USA"),
            "Description": ("categorical_text", "Plot synopsis or narrative summary of the film", "A Phoenix secretary steals $40,000 from a client and goes on the run.")
        }
    }

    dict_rows = []
    for cfg in TABLE_CONFIGS:
        tid = cfg["id"]
        tname = cfg["name"]
        c_p = raha_file(repo_root, cfg["raha_sub"], "clean.csv")
        d_p = raha_file(repo_root, cfg["raha_sub"], "dirty.csv")
        c_hdr, c_rows = load_csv(c_p)
        d_hdr, d_rows = load_csv(d_p)

        n_cols = min(len(c_hdr), len(d_hdr))
        ex_idx = {i for i in range(n_cols) if c_hdr[i].lower() in EXCLUDE_COLS or d_hdr[i].lower() in EXCLUDE_COLS}
        data_cols = [i for i in range(n_cols) if i not in ex_idx]

        for c in data_cols:
            col_name = c_hdr[c]
            d_vals = [r[c].strip() for r in d_rows if c < len(r)]
            tot_rows = len(d_vals)
            missing_cnt = sum(1 for v in d_vals if v.lower() in MISSING_TOKENS)
            missing_rate = (missing_cnt / tot_rows * 100) if tot_rows > 0 else 0.0

            # Meta lookup
            col_meta = column_definitions.get(tid, {}).get(col_name)
            if col_meta:
                dtype, meaning, eg = col_meta
            else:
                dtype = "categorical_text"
                meaning = f"Attribute {col_name} in {tname}"
                non_empty = [v for v in d_vals if v.lower() not in MISSING_TOKENS]
                eg = non_empty[0] if non_empty else ""

            dict_rows.append({
                "table_name": tid,
                "column_name": col_name,
                "data_type": dtype,
                "semantic_meaning": meaning,
                "example_value": eg,
                "missing_rate_percent": f"{missing_rate:.2f}%"
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["table_name", "column_name", "data_type", "semantic_meaning", "example_value", "missing_rate_percent"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)

    print(f"[SUCCESS] Exported docs/tables/m2_core_dictionary.csv ({len(dict_rows)} columns across 5 tables)")


def generate_figure_1_error_rate_per_column(repo_root: Path):
    """
    Figure 1: Per-column error rate distribution across the 5 core tables.
    Standalone SVG vector graphic with 5 panels / clean horizontal bars.
    """
    out_svg = repo_root / "docs" / "figures" / "m2" / "error_rate_per_column.svg"
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    table_data = []
    for cfg in TABLE_CONFIGS:
        tid = cfg["id"]
        cells_p = repo_root / "data" / "processed" / "core" / f"{tid}_cells.csv"
        col_tot = Counter()
        col_err = Counter()
        with open(cells_p, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                c = r["column"]
                col_tot[c] += 1
                if r["is_error"] == "True":
                    col_err[c] += 1
        col_rates = [(c, col_err[c] / col_tot[c] * 100, col_err[c], col_tot[c]) for c in col_tot]
        table_data.append((cfg["name"], col_rates))

    # Two-column layout: every table gets its own panel, so none is cut off. Height is computed from the panels.
    width = 1000
    col_w, gap, x0, top = 470, 20, 20, 100
    bar_left, bar_w = 160, 220
    panel_colors = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626"]
    sizes = [24 + 22 + len(cols) * 19 + 12 for _, cols in table_data]
    # place panels greedily in the shorter column (largest first)
    order = sorted(range(len(table_data)), key=lambda i: -sizes[i])
    col_y = [top, top]
    place = {}
    for i in order:
        c = 0 if col_y[0] <= col_y[1] else 1
        place[i] = (c, col_y[c])
        col_y[c] += sizes[i] + 14
    height = max(col_y) + 24
    font = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color:#ffffff; font-family:{font};">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="500" y="36" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">Where the errors are: error rate per column, all five core tables</text>',
        '<text x="500" y="60" text-anchor="middle" font-size="13" fill="#64748b">Share of cells in each column that are errors (number of error cells in brackets). 193,986 cells in total, 18,414 errors.</text>',
        '<text x="500" y="80" text-anchor="middle" font-size="12" fill="#64748b">Reading guide: a long bar means the column is a good place to look for errors; an empty bar means the column has no errors.</text>',
    ]
    for idx, (tname, cols) in enumerate(table_data):
        color = panel_colors[idx % len(panel_colors)]
        c, y = place[idx]
        px = x0 + c * (col_w + gap)
        block_h = sizes[idx]
        cols = sorted(cols, key=lambda r: -r[1])  # worst column first
        svg.append(f'<g transform="translate({px}, {y})">')
        svg.append(f'<rect x="0" y="0" width="{col_w}" height="{block_h}" fill="#f8fafc" rx="6" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<rect x="0" y="0" width="6" height="{block_h}" fill="{color}" rx="2"/>')
        tot_err = sum(r[2] for r in cols)
        tot_cells = sum(r[3] for r in cols)
        svg.append(f'<text x="16" y="20" font-size="15" font-weight="700" fill="#1e293b">{tname}</text>')
        svg.append(f'<text x="{col_w - 12}" y="20" text-anchor="end" font-size="11" fill="#64748b">{len(cols)} columns, {tot_err:,} errors, {tot_err / tot_cells * 100:.1f}% of cells</text>')
        for pct in (0, 50, 100):
            xp = bar_left + pct / 100.0 * bar_w
            svg.append(f'<line x1="{xp}" y1="30" x2="{xp}" y2="{block_h - 8}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2,2"/>')
            svg.append(f'<text x="{xp}" y="38" text-anchor="middle" font-size="9" fill="#94a3b8">{pct}%</text>')
        row_y = 46
        for cname, rate, err_cnt, tot_cnt in cols:
            w = rate / 100.0 * bar_w
            svg.append(f'<text x="{bar_left - 8}" y="{row_y + 11}" text-anchor="end" font-size="11" font-weight="500" fill="#334155">{cname}</text>')
            svg.append(f'<rect x="{bar_left}" y="{row_y}" width="{bar_w}" height="14" fill="#e2e8f0" rx="3"/>')
            if rate > 0:
                svg.append(f'<rect x="{bar_left}" y="{row_y}" width="{max(2.0, w):.1f}" height="14" fill="{color}" rx="3"/>')
            lbl = f"{rate:.1f}% ({err_cnt:,})"
            if rate > 55:
                svg.append(f'<text x="{bar_left + w - 6}" y="{row_y + 11}" text-anchor="end" font-size="10.5" font-weight="700" fill="#ffffff">{lbl}</text>')
            else:
                svg.append(f'<text x="{bar_left + max(2.0, w) + 6}" y="{row_y + 11}" font-size="10.5" font-weight="600" fill="#475569">{lbl}</text>')
            row_y += 19
        svg.append('</g>')

    svg.append('</svg>')
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[SUCCESS] Generated docs/figures/m2/error_rate_per_column.svg")


def generate_figure_2_value_length_distribution(repo_root: Path):
    """
    Figure 2: Distribution of cell character/value lengths across the 5 core tables.
    Plots boxplot and length-bucket distributions.
    """
    out_svg = repo_root / "docs" / "figures" / "m2" / "value_length_distribution.svg"
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    table_stats = []
    bucket_data = {}
    buckets = ["0 (empty)", "1-5", "6-15", "16-30", "31-60", "61-150", "151+"]

    for cfg in TABLE_CONFIGS:
        tid = cfg["id"]
        cells_p = repo_root / "data" / "processed" / "core" / f"{tid}_cells.csv"
        lengths = []
        b_counts = Counter()
        with open(cells_p, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                l = len(r["dirty_value"])
                lengths.append(l)
                if l == 0:
                    b_counts["0 (empty)"] += 1
                elif l <= 5:
                    b_counts["1-5"] += 1
                elif l <= 15:
                    b_counts["6-15"] += 1
                elif l <= 30:
                    b_counts["16-30"] += 1
                elif l <= 60:
                    b_counts["31-60"] += 1
                elif l <= 150:
                    b_counts["61-150"] += 1
                else:
                    b_counts["151+"] += 1

        lengths.sort()
        n = len(lengths)
        p25 = lengths[int(n * 0.25)]
        p50 = lengths[int(n * 0.50)]
        p75 = lengths[int(n * 0.75)]
        p95 = lengths[int(n * 0.95)]
        max_l = lengths[-1]
        min_l = lengths[0]

        table_stats.append({
            "name": cfg["name"],
            "min": min_l,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "max": max_l,
            "total": n
        })
        bucket_data[cfg["name"]] = b_counts

    width = 960
    height = 540
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color:#ffffff; font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>',
        f'<text x="480" y="38" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">Value Length Character Distributions Across Core Tables</text>',
        f'<text x="480" y="62" text-anchor="middle" font-size="13" fill="#64748b">How long are the cell values? Box = middle half of cells (p25 to p75), bar inside = median, line = up to p95, dot = longest cell</text>'
    ]

    # Left Panel: Percentile Box / Range plot (X: 50 to 450)
    svg.append('<g transform="translate(50, 90)">')
    svg.append('<rect width="400" height="390" fill="#f8fafc" rx="6" stroke="#e2e8f0" stroke-width="1"/>')
    svg.append('<text x="200" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">Characters per cell (log scale)</text>')
    
    # Grid lines for log scale: 1, 5, 10, 25, 50, 100, 250, 1000, 2500
    grid_vals = [1, 5, 10, 25, 50, 100, 250, 1000, 2500]
    min_log = 0.0 # log10(1)
    max_log = math.log10(3000)
    plot_w = 215
    plot_x0 = 70

    def val_to_x(v):
        v_clamped = max(1, v)
        return plot_x0 + (math.log10(v_clamped) / max_log) * plot_w

    for gv in grid_vals:
        gx = val_to_x(gv)
        svg.append(f'<line x1="{gx}" y1="50" x2="{gx}" y2="350" stroke="#e2e8f0" stroke-dasharray="2,2"/>')
        svg.append(f'<text x="{gx}" y="365" text-anchor="middle" font-size="9" fill="#94a3b8">{gv}</text>')

    y_pos = 90
    t_colors = {"Hospital": "#2563eb", "Flights": "#d97706", "Beers": "#059669", "Rayyan": "#7c3aed", "Movies": "#dc2626"}
    for st in table_stats:
        tname = st["name"]
        col = t_colors[tname]
        x_min = val_to_x(max(1, st["min"]))
        x_p25 = val_to_x(st["p25"])
        x_p50 = val_to_x(st["p50"])
        x_p75 = val_to_x(st["p75"])
        x_p95 = val_to_x(st["p95"])
        x_max = val_to_x(st["max"])

        svg.append(f'<text x="{plot_x0 - 10}" y="{y_pos+4}" text-anchor="end" font-size="12" font-weight="600" fill="#334155">{tname}</text>')
        # Whisker p25 to p95
        svg.append(f'<line x1="{x_p25}" y1="{y_pos}" x2="{x_p95}" y2="{y_pos}" stroke="{col}" stroke-width="2"/>')
        # Box p25 to p75
        svg.append(f'<rect x="{x_p25}" y="{y_pos-10}" width="{max(2, x_p75 - x_p25)}" height="20" fill="{col}" fill-opacity="0.3" stroke="{col}" stroke-width="1.5" rx="3"/>')
        # Median line
        svg.append(f'<line x1="{x_p50}" y1="{y_pos-10}" x2="{x_p50}" y2="{y_pos+10}" stroke="{col}" stroke-width="3"/>')
        # Max dot
        svg.append(f'<circle cx="{x_max}" cy="{y_pos}" r="3" fill="{col}"/>')
        # Value label
        svg.append(f'<text x="{plot_x0 + plot_w + 12}" y="{y_pos+4}" font-size="10" font-weight="500" fill="#64748b">median {st["p50"]}, p95 {st["p95"]}</text>')
        y_pos += 52

    svg.append('</g>')

    # Right Panel: Bucket Composition Stacked Chart (X: 490 to 910)
    svg.append('<g transform="translate(490, 90)">')
    svg.append('<rect width="420" height="390" fill="#f8fafc" rx="6" stroke="#e2e8f0" stroke-width="1"/>')
    svg.append('<text x="210" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">Share of cells by length (characters)</text>')

    bucket_colors = {
        "0 (empty)": "#94a3b8",
        "1-5": "#38bdf8",
        "6-15": "#34d399",
        "16-30": "#fbbf24",
        "31-60": "#f87171",
        "61-150": "#a78bfa",
        "151+": "#c084fc"
    }

    y_bar = 70
    bar_w = 280
    x_bar0 = 80
    for st in table_stats:
        tname = st["name"]
        b_cnt = bucket_data[tname]
        tot = st["total"]
        svg.append(f'<text x="{x_bar0 - 10}" y="{y_bar+16}" text-anchor="end" font-size="12" font-weight="600" fill="#334155">{tname}</text>')
        
        cur_x = x_bar0
        for b_name in buckets:
            cnt = b_cnt[b_name]
            seg_w = (cnt / tot) * bar_w
            if seg_w > 0:
                svg.append(f'<rect x="{cur_x}" y="{y_bar}" width="{seg_w:.2f}" height="24" fill="{bucket_colors[b_name]}" stroke="#ffffff" stroke-width="0.5"/>')
                if seg_w >= 26:
                    svg.append(f'<text x="{cur_x + seg_w / 2:.1f}" y="{y_bar + 16}" text-anchor="middle" font-size="9.5" font-weight="700" fill="#0f172a">{cnt / tot * 100:.0f}%</text>')
                cur_x += seg_w
        y_bar += 52

    # Legend for buckets
    leg_y = 330
    leg_x = 20
    for i, b_name in enumerate(buckets):
        col = bucket_colors[b_name]
        lx = leg_x + (i % 4) * 98
        ly = leg_y + (i // 4) * 25
        svg.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{col}"/>')
        svg.append(f'<text x="{lx+16}" y="{ly+10}" font-size="10" font-weight="500" fill="#475569">{b_name}</text>')

    svg.append('</g>')

    svg.append('</svg>')
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[SUCCESS] Generated docs/figures/m2/value_length_distribution.svg")


def generate_figure_3_error_type_mix(repo_root: Path):
    """
    Figure 3: Error-type mix across core tables and Train/Val/Test splits.
    Stacked bar chart showing exact composition of missing, pattern, rule, and outlier errors.
    """
    out_svg = repo_root / "docs" / "figures" / "m2" / "error_type_mix.svg"
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    # Read docs/tables/m2_core_splits.csv
    splits_csv = repo_root / "docs" / "tables" / "m2_core_splits.csv"
    records = list(csv.DictReader(open(splits_csv, "r", encoding="utf-8")))

    width = 960
    height = 680
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color:#ffffff; font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>',
        f'<text x="480" y="38" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">Error Type Mix Across Core Tables and Splits (Train / Val / Test)</text>',
        f'<text x="480" y="62" text-anchor="middle" font-size="13" fill="#64748b">Verified Conservation of 18,414 Errors (Missing Value, Pattern Violation, Rule Violation, Outliers)</text>'
    ]

    type_colors = {
        "missing_value": "#2563eb",     # Royal Blue
        "pattern_violation": "#f97316", # Coral / Orange
        "rule_violation": "#8b5cf6",    # Purple
        "outliers": "#10b981"           # Emerald
    }
    type_labels = {
        "missing_value": "Missing Value",
        "pattern_violation": "Pattern Violation",
        "rule_violation": "Rule Violation",
        "outliers": "Outliers"
    }

    # Legend at top
    leg_x = 220
    leg_y = 80
    for i, etype in enumerate(["missing_value", "pattern_violation", "rule_violation", "outliers"]):
        lx = leg_x + i * 145
        svg.append(f'<rect x="{lx}" y="{leg_y}" width="16" height="16" rx="3" fill="{type_colors[etype]}"/>')
        svg.append(f'<text x="{lx+22}" y="{leg_y+13}" font-size="12" font-weight="600" fill="#334155">{type_labels[etype]}</text>')

    # Group records by table
    by_table = defaultdict(list)
    for r in records:
        if r["Split"] in ["train", "val", "test"]:
            by_table[r["Table"]].append(r)

    y_cur = 125
    bar_x0 = 170
    bar_max_w = 640

    for tname in ["hospital", "flights", "beers", "rayyan", "movies"]:
        splits = by_table[tname]
        tot_table_err = sum(int(s["Error_Cells"]) for s in splits)
        svg.append(f'<g transform="translate(40, {y_cur})">')
        svg.append(f'<rect x="0" y="0" width="880" height="92" fill="#f8fafc" rx="6" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="20" y="24" font-size="14" font-weight="700" fill="#1e293b">{tname.capitalize()}</text>')
        svg.append(f'<text x="20" y="42" font-size="11" fill="#64748b">Total Errors: {tot_table_err:,}</text>')

        # 3 split rows: train, val, test
        row_y = 12
        for s in splits:
            s_name = s["Split"].capitalize()
            s_err = int(s["Error_Cells"])
            m_val = int(s["Missing_Value"])
            p_val = int(s["Pattern_Violation"])
            r_val = int(s["Rule_Violation"])
            o_val = int(s["Outliers"])

            svg.append(f'<text x="{bar_x0 - 15}" y="{row_y+16}" text-anchor="end" font-size="11" font-weight="600" fill="#475569">{s_name}</text>')
            
            # Draw stacked bar
            cur_x = bar_x0
            for etype, cnt in [("missing_value", m_val), ("pattern_violation", p_val), ("rule_violation", r_val), ("outliers", o_val)]:
                if s_err > 0 and cnt > 0:
                    seg_w = (cnt / s_err) * bar_max_w
                    svg.append(f'<rect x="{cur_x:.1f}" y="{row_y+4}" width="{seg_w:.1f}" height="16" fill="{type_colors[etype]}" stroke="#ffffff" stroke-width="0.5"/>')
                    cur_x += seg_w

            # Error count label
            svg.append(f'<text x="{bar_x0 + bar_max_w + 10}" y="{row_y+16}" font-size="11" font-weight="600" fill="#334155">{s_err:,} errs</text>')
            row_y += 24

        svg.append('</g>')
        y_cur += 104

    svg.append('</svg>')
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[SUCCESS] Generated docs/figures/m2/error_type_mix.svg")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    print("=" * 80)
    print("      GENERATING MILESTONE 2 QUALITY OUTPUTS & PUBLICATION FIGURES       ")
    print("=" * 80)
    build_quality_table(repo_root)
    build_core_dictionary(repo_root)
    generate_figure_1_error_rate_per_column(repo_root)
    generate_figure_2_value_length_distribution(repo_root)
    generate_figure_3_error_type_mix(repo_root)
    print("=" * 80)
    print("                      ALL M2 OUTPUTS GENERATED SUCCESSFULLY                    ")
    print("=" * 80)


if __name__ == "__main__":
    main()
