"""M4 inference options of the context builder, the repair confidence and the stage 4 gate."""
import pandas as pd

from scripts.run_pipeline import gate_decision
from src.llm.infer import repair_confidence
from src.profiling.context import TableContext
from tests.test_m4_detectors import multi_source_flights


def test_training_prompts_unchanged_by_default():
    # flight -> dep holds for 10 of 12 rows (0.83 < 0.9), so with the training defaults no flight evidence is shown
    facts = TableContext("f", multi_source_flights()).facts(0, "dep")
    assert all(d["col"] != "flight" for d in facts["determined_by"])


def test_entity_column_is_found_and_shown_first():
    ctx = TableContext("f", multi_source_flights(), evidence_threshold=0.5, skip_missing_keys=True,
                       weighted_majority=True)
    assert ctx.entity_key() == "flight"
    first = ctx.facts(0, "dep")["determined_by"][0]
    assert first["col"] == "flight" and first["majority"] == "7:53 a.m."


def test_no_entity_column_with_one_row_per_object():
    df = pd.DataFrame({"id": list("abcdef"), "v": list("123456")})
    assert TableContext("t", df).entity_key() is None


def test_repair_confidence_uses_only_the_repair_value_tokens():
    class FakeTokenizer:
        pieces = {0: ' true, "repair": "', 1: "10", 2: ":30", 3: ' p.m."}'}

        def decode(self, ids, skip_special_tokens=True):
            return "".join(self.pieces[i] for i in ids)

    tok = FakeTokenizer()
    conf = repair_confidence(tok, [0, 1, 2, 3], [0.99, 0.9, 0.3, 0.99], tok.decode([0, 1, 2, 3]))
    assert abs(conf["min"] - 0.3) < 1e-9
    assert repair_confidence(tok, [0], [0.9], ' true, "repair": null}') is None


def test_gate_needs_a_confident_repair_value():
    pred = {"valid_json": True, "is_error": True, "p_error": 0.95, "repair": "12", "p_repair_min": 0.5}
    assert gate_decision(pred, 0.5, 0.8) == "repair"                       # confidence gate off
    assert gate_decision(pred, 0.5, 0.8, repair_conf=0.9) == "flag for review"
    assert gate_decision({**pred, "p_error": 0.3}, 0.5, 0.8) == "keep"
    assert gate_decision({**pred, "valid_json": False}, 0.5, 0.8) == "review (invalid answer)"
