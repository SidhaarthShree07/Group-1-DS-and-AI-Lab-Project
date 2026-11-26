#!/usr/bin/env python3
"""Repair validation layer (stage 4b, added in M4): checks a proposed repair against independent rules first.

Author: Sidhaarth Shree

Place in the pipeline: the stage 4 gate says "repair" -> this layer -> the repair is applied, or the cell goes to
review.

For every repair the gate accepts, the layer runs four kinds of checks and writes an audit record:
  1. value checks: data type, format (pattern), permitted values and numeric range of the proposed value;
  2. business rules: rules of the column from an independent source (configs/validation/<table>.yaml);
  3. relations: functional dependencies, computed columns and unique keys between the affected cell and the other
     columns of its row and the other rows of the table;
  4. before/after consistency: the number of rule violations of the affected record is counted with the old and with
     the proposed value.

The rules come from domain knowledge, public references or the definition of the dataset, and every rule names its
source. They are never learned from the table being cleaned. The data of the table is only used as evidence when a
rule is applied (for example: which other rows share the provider number).

Outcomes (the original value is always kept in the change log, so every repair can be reverted):
  validated      every applicable check passes and the record is not less consistent than before -> repair applied
  rejected       the proposed value fails a check, or it adds an inconsistency -> referred for review
  unvalidated    rules cover the column, but the old value breaks none of them, so the repair cannot be confirmed
                 -> referred for review
  not covered    no rule covers the column (or the table has no rule file) -> the layer cannot check it; by default
                 referred for review, or applied on the gate's confidence alone with apply_unverified=True
  no change      the proposed value equals the old value -> not a repair; the cell is kept

Every cell also gets a state for the user (repair_state): "verified" (applied and passed the rules), "not verified"
(applied on model confidence alone), "needs review" (not applied) or "unchanged".

Passing means the value is consistent with the rules we have. It is not proof that the value is correct.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

US_STATES = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me",
    "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa",
    "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc", "pr", "gu", "vi", "as", "mp",
]  # USPS codes: 50 states, District of Columbia, 5 territories
REFERENCES = {"us_states": US_STATES}
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "validation"


def clean(value) -> str:
    """Comparison form: strip and collapse spaces (case is kept; rules say when to ignore it)."""
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def rule_label(rel: dict) -> str:
    if rel["kind"] == "fd":
        return f"{'+'.join(rel['lhs'])} -> {'+'.join(rel['rhs'])}"
    if rel["kind"] == "concat":
        return f"{rel['target']} = " + " + ".join(repr(p) if p in ("_", "-", " ") else p for p in rel["parts"])
    return f"unique {'+'.join(rel.get('columns', []))}"


def load_rules(table: str, config_dir: Path = CONFIG_DIR) -> Optional[dict]:
    path = Path(config_dir) / f"{table}.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class Check:
    name: str
    kind: str          # value | business | relation | consistency
    source: str
    before: Optional[bool]   # did the old value pass (None = not applicable)
    after: Optional[bool]    # does the proposed value pass
    detail: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Result:
    table: str
    row: int
    column: str
    old: str
    new: str
    outcome: str
    reason: str
    violations_before: int = 0
    violations_after: int = 0
    checks: List[Check] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "checks"}
        d["checks"] = [c.as_dict() for c in self.checks]
        return d


class RepairValidator:
    """Validates proposed repairs for one table. `dirty` is the table as the pipeline received it."""

    def __init__(self, table: str, dirty: pd.DataFrame, rules: Optional[dict] = None):
        self.table = table
        self.df = dirty.reset_index(drop=True).astype(str)
        self.low = self.df.map(lambda x: clean(x).lower())  # comparison copy for the relation checks
        self.rules = rules if rules is not None else (load_rules(table) or {"columns": {}, "relations": []})
        self.missing = {clean(t).lower() for t in self.rules.get("missing_tokens", [""])}
        self.cols = self.rules.get("columns", {}) or {}
        self.rels = [r for r in (self.rules.get("relations", []) or [])
                     if all(c in self.df.columns for c in self._rel_columns(r))]
        self.cols = {c: spec for c, spec in self.cols.items() if c in self.df.columns}

    # ---- single-value checks ------------------------------------------------------------------------------------
    def _is_missing(self, v: str) -> bool:
        return clean(v).lower() in self.missing or clean(v).lower() in {"null", "none", "nan"}

    def value_checks(self, column: str, value: str) -> List[tuple]:
        """[(name, kind, source, passed, detail)] for one value of one column."""
        spec = self.cols.get(column)
        if not spec:
            return []
        src, v, out = spec.get("source", ""), clean(value), []
        if self._is_missing(v):
            ok = bool(spec.get("missing_ok", False))
            return [("missing allowed" if ok else "value required", "value", src, ok, f"value {v!r} is missing")]
        if "pattern" in spec:
            out.append(("format", "value", src, re.fullmatch(spec["pattern"], v) is not None, f"pattern {spec['pattern']}"))
        if "allowed" in spec or "allowed_ref" in spec:
            allowed = list(spec.get("allowed", [])) + list(REFERENCES.get(spec.get("allowed_ref", ""), []))
            up = spec.get("case") == "upper"  # upper: the value must be written in capitals; otherwise compare in lower case
            vals = {a.upper() if up else a.lower() for a in allowed}
            out.append(("permitted value", "business", src, (v if up else v.lower()) in vals,
                        f"{len(vals)} permitted values"))
        if "range" in spec:
            r = spec["range"]
            raw = v.replace(r["strip"], "") if r.get("strip") else v
            try:
                x = float(raw)
                ok = r.get("min", float("-inf")) <= x <= r.get("max", float("inf"))
                out.append(("range", "value", src, ok, f"{x:g} in [{r.get('min')}, {r.get('max')}]"))
            except ValueError:
                out.append(("numeric type", "value", src, False, f"{v!r} is not a number"))
        return out

    # ---- relational checks --------------------------------------------------------------------------------------
    @staticmethod
    def _rel_columns(rel: dict) -> List[str]:
        if rel["kind"] == "fd":
            return list(rel["lhs"]) + list(rel["rhs"])
        if rel["kind"] == "concat":
            return [rel["target"]] + [p for p in rel["parts"] if p not in ("_", "-", " ")]
        return list(rel.get("columns", []))

    def _row(self, row: int, column: str, value: str) -> Dict[str, str]:
        r = self.df.iloc[row].to_dict()
        r[column] = value
        return r

    def _fd_violations(self, rel: dict, row: int, record: Dict[str, str], only: Optional[str] = None) -> int:
        """Right-hand-side columns on which the record disagrees with the other rows that share its left-hand side.
        The rule (A determines B) is independent; the other rows are the evidence: the record should carry B's most
        common value in its A group (ties: any of the most common values). A group of one row gives no evidence."""
        key = [clean(record[c]).lower() for c in rel["lhs"]]
        if any(self._is_missing(k) for k in key):
            return 0
        mask = pd.Series(True, index=self.low.index)
        for c, k in zip(rel["lhs"], key):
            mask &= self.low[c] == k
        mask.iloc[row] = False
        others = self.low[mask]
        if others.empty:
            return 0
        # truth discovery: each related row counts with its reliability, the share of its other columns that agree
        # with the group (multi-source data: low-quality sources that copy each other must not outvote the truth;
        # Li et al., PVLDB 2012; the same weighting as the prompt's evidence, src/profiling/context.py)
        cols = [c for c in self.low.columns if c not in rel["lhs"]]
        group_majority = {c: others[c].value_counts().index[0] for c in cols}
        violations = 0
        for b in (rel["rhs"] if only is None else [only]):
            rest = [c for c in cols if c != b]
            weights = others[rest].eq(pd.Series(group_majority)[rest]).sum(axis=1).add(1).div(len(rest) + 2)
            votes = weights.groupby(others[b]).sum()
            if len(votes) > 1:
                votes = votes[[value not in self.missing for value in votes.index]]
            best_values = set(votes[votes >= votes.max() - 1e-9].index) if len(votes) else set()
            violations += int(clean(record[b]).lower() not in best_values)
        return violations

    def relation_violations(self, rel: dict, row: int, record: Dict[str, str]) -> int:
        if rel["kind"] == "fd":
            return self._fd_violations(rel, row, record)
        if rel["kind"] == "concat":
            parts = [p if p in ("_", "-", " ") else clean(record[p]) for p in rel["parts"]]
            return int(clean(record[rel["target"]]).lower() != "".join(parts).lower())
        if rel["kind"] == "unique":
            key = tuple(clean(record[c]).lower() for c in rel["columns"])
            mask = pd.Series(True, index=self.low.index)
            for c, k in zip(rel["columns"], key):
                mask &= self.low[c] == k
            mask.iloc[row] = False
            return int(mask.sum())
        return 0

    def record_violations(self, row: int, record: Dict[str, str]) -> int:
        """All rule violations of one record: failed value checks of every column plus relation violations."""
        n = sum(1 for c in self.cols for (_, _, _, ok, _) in self.value_checks(c, record[c]) if not ok)
        return n + sum(self.relation_violations(rel, row, record) for rel in self.rels)

    # ---- the decision -------------------------------------------------------------------------------------------
    def validate(self, row: int, column: str, new) -> Result:
        t0 = time.perf_counter()
        old = clean(self.df.at[row, column])
        new = "" if new is None else clean(new)
        res = Result(self.table, int(row), column, old, new, "", "")
        if old == new:
            res.outcome, res.reason = "no change", "the proposed value equals the old value"
            res.seconds = time.perf_counter() - t0
            return res
        before, after = self._row(row, column, old), self._row(row, column, new)

        old_checks = self.value_checks(column, old)
        old_ok = all(o for (_, _, _, o, _) in old_checks) if old_checks else None
        for (name, kind, src, ok_new, detail) in self.value_checks(column, new):
            res.checks.append(Check(name, kind, src, old_ok, ok_new, detail))
        relation_violations_before = 0
        for rel in self.rels:
            if column not in self._rel_columns(rel):
                continue
            label = rule_label(rel)
            if rel["kind"] == "fd" and column in rel["rhs"]:
                # the repaired cell itself must agree with the related rows (same left-hand side), when there are any
                n_before = self._fd_violations(rel, row, before, only=column)
                n_after = self._fd_violations(rel, row, after, only=column)
                ok, what = n_after == 0, "the cell disagrees with the most common value of its related rows"
            else:
                n_before = self.relation_violations(rel, row, before)
                n_after = self.relation_violations(rel, row, after)
                ok, what = n_after <= n_before, "violations with other cells or rows"
            relation_violations_before += n_before
            res.checks.append(Check(f"{rel['kind']}: {label}", "relation", rel.get("source", ""), n_before == 0, ok,
                                    f"{what}: {n_before} before, {n_after} after"))
        res.violations_before = self.record_violations(row, before)
        res.violations_after = self.record_violations(row, after)
        res.checks.append(Check("record consistency", "consistency", "all rules of this table", None,
                                res.violations_after <= res.violations_before,
                                f"record violations: {res.violations_before} before, {res.violations_after} after"))

        rule_checks = [c for c in res.checks if c.kind != "consistency"]
        failed = [c for c in res.checks if c.after is False]
        # evidence that the old value is wrong: it fails a value check, or the record breaks a rule through this cell
        evidence = old_ok is False or relation_violations_before > 0
        if not rule_checks:
            res.outcome = "not covered"
            res.reason = f"no independent rule covers column {column!r}; the repair cannot be validated"
        elif failed:
            res.outcome = "rejected"
            res.reason = "; ".join(f"{c.name} ({c.detail})" for c in failed)
        elif not evidence:
            res.outcome = "unvalidated"
            res.reason = ("the old value already satisfies every independent rule; with no inconsistency to resolve, "
                          "the repair cannot be confirmed")
        elif res.violations_after >= res.violations_before:
            res.outcome = "rejected"
            res.reason = (f"the repair does not resolve the inconsistency (record violations {res.violations_before} "
                          f"before, {res.violations_after} after)")
        else:
            res.outcome = "validated"
            res.reason = f"{len(rule_checks)} checks passed (consistency validation, not proof of correctness)"
        res.seconds = time.perf_counter() - t0
        return res

    def apply(self, row: int, column: str, value: str) -> None:
        """Make a validated repair visible to later checks (relations are then judged on the repaired table)."""
        self.df.at[row, column] = clean(value)
        self.low.at[row, column] = clean(value).lower()


def validate_predictions(table: str, dirty: pd.DataFrame, preds: List[dict], rules: Optional[dict] = None,
                         apply_unverified: bool = False) -> List[Result]:
    """Validate every gate repair of one table, most confident first; validated repairs are applied as they pass, so
    later repairs are judged on the repaired table. Adds `gate_decision`, `validation`, the final `decision` and the
    user-facing `state` to each prediction (in place) and returns the audit records. With apply_unverified, a repair
    in a column no rule covers keeps the gate's decision and is marked "not verified"."""
    validator = RepairValidator(table, dirty, rules)
    results = []
    gate_repairs = [p for p in preds if p.get("decision") == "repair"]
    for p in sorted(gate_repairs, key=lambda p: -(p.get("p_error") or 0.0)):
        result = validator.validate(int(p["row"]), p["column"], p.get("repair"))
        if result.outcome == "validated":
            validator.apply(result.row, result.column, result.new)
        p["validation"] = {"outcome": result.outcome, "reason": result.reason}
        results.append(result)
    for p in preds:
        p.setdefault("gate_decision", p.get("decision"))
        outcome = (p.get("validation") or {}).get("outcome")
        p["decision"] = final_decision(p["gate_decision"], outcome, apply_unverified)
        p["state"] = repair_state(p["decision"], outcome)
    return results


def final_decision(gate_decision: str, outcome: Optional[str], apply_unverified: bool = False) -> str:
    """The decision after the layer. Only gate repairs are validated; everything else passes through."""
    if gate_decision != "repair" or outcome is None:
        return gate_decision
    if outcome == "not covered" and apply_unverified:
        return "repair"
    return {"validated": "repair", "no change": "keep", "rejected": "review (repair rejected by validation)",
            "unvalidated": "review (repair cannot be validated)",
            "not covered": "review (repair cannot be validated)"}[outcome]


def repair_state(decision: str, outcome: Optional[str] = None) -> str:
    """What the user is told about a cell: an applied repair is "verified" only if the rules checked it."""
    if decision == "repair":
        return "verified" if outcome == "validated" else "not verified"
    if decision.startswith("review") or decision == "flag for review":
        return "needs review"
    return "unchanged"
