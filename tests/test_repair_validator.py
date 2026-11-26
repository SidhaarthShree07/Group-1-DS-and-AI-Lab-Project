"""Repair validation layer (src/validation/repair_validator.py) on small hand-made tables."""
import pandas as pd

from src.validation.repair_validator import RepairValidator, final_decision, load_rules, repair_state, validate_predictions

RULES = {
    "missing_tokens": ["empty"],
    "columns": {
        "zip": {"pattern": r"^\d{5}$", "source": "USPS"},
        "state": {"allowed_ref": "us_states", "source": "USPS"},
        "score": {"pattern": r"^\d{1,3}%$", "range": {"min": 0, "max": 100, "strip": "%"}, "missing_ok": True,
                  "source": "percentage"},
    },
    "relations": [
        {"kind": "fd", "lhs": ["pid"], "rhs": ["city", "zip"], "source": "one provider, one address"},
        {"kind": "concat", "target": "key", "parts": ["state", "_", "code"], "source": "definition"},
    ],
}


def table():
    return pd.DataFrame({
        "pid": ["1", "1", "1", "2"],
        "city": ["dothan", "dothan", "dxthan", "sheffield"],
        "zip": ["36302", "36302", "3630x", "35660"],
        "state": ["al", "al", "al", "al"],
        "code": ["hf-1", "hf-2", "hf-3", "hf-1"],
        "key": ["al_hf-1", "al_hf-2", "al_hf-3", "al_hf-1"],
        "score": ["90%", "80%", "70%", "empty"],
        "note": ["a", "b", "c", "d"],
    })


def test_validated_repair_resolves_inconsistency():
    r = RepairValidator("t", table(), RULES).validate(2, "zip", "36302")
    assert r.outcome == "validated"
    assert r.violations_after < r.violations_before
    assert any(c.kind == "relation" for c in r.checks) and any(c.kind == "value" for c in r.checks)


def test_rejected_bad_format_and_range():
    v = RepairValidator("t", table(), RULES)
    assert v.validate(2, "zip", "3630").outcome == "rejected"           # format
    assert v.validate(0, "score", "120%").outcome == "rejected"         # range
    assert v.validate(0, "state", "zz").outcome == "rejected"           # permitted values (USPS list)


def test_rejected_when_it_disagrees_with_related_rows():
    r = RepairValidator("t", table(), RULES).validate(2, "city", "dotham")
    assert r.outcome == "rejected"
    assert "disagrees" in r.reason or "violations" in r.reason


def test_unvalidated_without_rule_or_without_evidence():
    v = RepairValidator("t", table(), RULES)
    assert v.validate(0, "note", "x").outcome == "not covered"          # no rule covers the column
    assert v.validate(0, "score", "95%").outcome == "unvalidated"       # old value already satisfies every rule


def test_deleting_a_valid_value_is_not_validated():
    r = RepairValidator("t", table(), RULES).validate(0, "score", "empty")
    assert r.outcome != "validated"


def test_no_change_and_final_decision():
    v = RepairValidator("t", table(), RULES)
    assert v.validate(0, "zip", "36302").outcome == "no change"
    assert final_decision("repair", "validated") == "repair"
    assert final_decision("repair", "rejected").startswith("review")
    assert final_decision("repair", "unvalidated").startswith("review")
    assert final_decision("repair", "not covered").startswith("review")          # default: review
    assert final_decision("repair", "not covered", apply_unverified=True) == "repair"
    assert final_decision("repair", "unvalidated", apply_unverified=True).startswith("review")  # rules exist
    assert final_decision("repair", "no change") == "keep"
    assert final_decision("flag for review", None) == "flag for review"


def test_validate_predictions_keeps_gate_decision_and_audit():
    preds = [
        {"row": 2, "column": "zip", "repair": "36302", "p_error": 0.9, "decision": "repair"},
        {"row": 2, "column": "city", "repair": "dotham", "p_error": 0.95, "decision": "repair"},
        {"row": 0, "column": "note", "repair": None, "p_error": 0.6, "decision": "flag for review"},
    ]
    results = validate_predictions("t", table(), preds, RULES)
    assert len(results) == 2
    assert [p["gate_decision"] for p in preds] == ["repair", "repair", "flag for review"]
    assert preds[0]["decision"] == "repair" and preds[1]["decision"].startswith("review")
    assert all(r.checks for r in results)


def test_shipped_rule_files_load():
    for t in ("hospital", "flights", "beers"):
        rules = load_rules(t)
        assert rules["columns"] and rules["relations"]
        for spec in rules["columns"].values():
            assert spec.get("source")
        for rel in rules["relations"]:
            assert rel.get("source")


def test_three_states_for_the_user():
    assert repair_state("repair", "validated") == "verified"
    assert repair_state("repair", "not covered") == "not verified"
    assert repair_state("repair", None) == "not verified"            # layer not run: confidence only
    assert repair_state("review (repair rejected by validation)", "rejected") == "needs review"
    assert repair_state("flag for review") == "needs review"
    assert repair_state("keep") == "unchanged"


def test_table_without_rules_applies_unverified_only_when_asked():
    dirty = pd.DataFrame({"a": ["x", "y"]})
    preds = [{"row": 0, "column": "a", "repair": "z", "decision": "repair", "p_error": 0.99}]
    default, unverified = [dict(preds[0])], [dict(preds[0])]
    validate_predictions("t", dirty, default, rules={"columns": {}, "relations": []})
    validate_predictions("t", dirty, unverified, rules={"columns": {}, "relations": []}, apply_unverified=True)
    assert default[0]["decision"].startswith("review") and default[0]["state"] == "needs review"
    assert unverified[0]["decision"] == "repair" and unverified[0]["state"] == "not verified"
