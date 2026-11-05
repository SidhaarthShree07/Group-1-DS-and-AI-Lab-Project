#!/usr/bin/env python3
"""Output parser for model-generated predictions.

Author: Sahilpreet

Converts raw text returned by the language model into structured prediction fields:
- is_error: bool
- error_type: str ("missing", "pattern", "rule", "outlier", "none")
- evidence: str
- repair: Optional[str]
- valid_json: bool

Follows the Milestone 3 specification and reuse contract:
- Reuses the canonical Raha normalisation rule from src.profiling.context.norm
- Robustly handles markdown code fences (```json ... ``` or ``` ... ```)
- Validates field types and allowed error types
- Rejects malformed JSON, missing fields, wrong types, and contradictory predictions:
  * is_error=False with error_type != "none" or repair != None
  * is_error=True with error_type == "none"
- For any invalid response or schema mismatch, returns a safe not-flagged prediction:
  is_error=False, error_type="none", evidence="", repair=None, valid_json=False
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root is on sys.path for direct module imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.profiling.context import norm

ALLOWED_ERROR_TYPES = frozenset({"missing", "pattern", "rule", "outlier", "none"})
REQUIRED_KEYS = frozenset({"is_error", "error_type", "evidence", "repair"})


def normalise_repair(repair: Optional[str]) -> Optional[str]:
    """Normalise a repair string using the canonical Raha comparison rule.

    Unescapes HTML entities, collapses whitespace runs into a single space,
    and strips leading and trailing whitespace. Returns None if repair is None.
    """
    if repair is None:
        return None
    return norm(repair)


def _safe_invalid_dict() -> Dict[str, Any]:
    """Return the safe fallback dictionary for invalid JSON or schema violations."""
    return {
        "is_error": False,
        "error_type": "none",
        "evidence": "",
        "repair": None,
        "valid_json": False,
    }


def _extract_json_text(text: str) -> str:
    """Extract candidate JSON text, stripping Markdown code fences if present."""
    text = text.strip()

    # Check for Markdown code block: ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json|JSON)?\s*(\{[\s\S]*?\})\s*```")
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()

    fence_generic = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")
    match_generic = fence_generic.search(text)
    if match_generic:
        candidate = match_generic.group(1).strip()
        if candidate:
            return candidate

    return text


def parse_prediction(
    raw_text: Optional[str],
    normalise_repair_value: bool = False,
) -> Dict[str, Any]:
    """Parse raw text from model generation into a structured prediction dictionary.

    Args:
        raw_text: The raw output string produced by the model (or None).
        normalise_repair_value: If True, applies Raha normalisation (norm) to
            a non-empty repair string. Defaults to False to preserve the model's
            exact textual output in accordance with M3 separation of concerns.

    Returns:
        A dictionary with keys:
            - is_error (bool): Flag indicating if cell is classified as an error.
            - error_type (str): "missing" | "pattern" | "rule" | "outlier" | "none".
            - evidence (str): Short explanation provided by the model.
            - repair (Optional[str]): Suggested corrected value, or None.
            - valid_json (bool): True if response strictly parsed and validated,
              False otherwise.
    """
    if raw_text is None or not isinstance(raw_text, str):
        return _safe_invalid_dict()

    candidate = _extract_json_text(raw_text)
    if not candidate:
        return _safe_invalid_dict()

    try:
        obj = json.loads(candidate)
    except Exception:
        # If direct json.loads fails, check for a balanced outer JSON object
        brace_start = candidate.find("{")
        brace_end = candidate.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                obj = json.loads(candidate[brace_start : brace_end + 1])
            except Exception:
                return _safe_invalid_dict()
        else:
            return _safe_invalid_dict()

    # Must be a JSON object (dict)
    if not isinstance(obj, dict):
        return _safe_invalid_dict()

    # Must contain all required keys
    if not REQUIRED_KEYS.issubset(obj.keys()):
        return _safe_invalid_dict()

    raw_is_error = obj["is_error"]
    raw_error_type = obj["error_type"]
    raw_evidence = obj["evidence"]
    raw_repair = obj["repair"]

    # Validate field types strictly
    if type(raw_is_error) is not bool:
        return _safe_invalid_dict()

    if not isinstance(raw_error_type, str):
        return _safe_invalid_dict()

    if not isinstance(raw_evidence, str):
        return _safe_invalid_dict()

    if raw_repair is not None and not isinstance(raw_repair, str):
        return _safe_invalid_dict()

    error_type = raw_error_type.strip().lower()
    if error_type not in ALLOWED_ERROR_TYPES:
        return _safe_invalid_dict()

    # Validate logical consistency between is_error, error_type, and repair
    if not raw_is_error:
        # A response with is_error=False must have error_type="none" and no repair.
        # Contradictions (e.g. error_type != "none" or repair string given) invalidate the schema.
        if error_type != "none":
            return _safe_invalid_dict()
        if raw_repair is not None and raw_repair != "":
            return _safe_invalid_dict()

        return {
            "is_error": False,
            "error_type": "none",
            "evidence": raw_evidence,
            "repair": None,
            "valid_json": True,
        }
    else:
        # A response with is_error=True cannot have error_type="none".
        if error_type == "none":
            return _safe_invalid_dict()

        repair = raw_repair
        if normalise_repair_value and repair is not None:
            repair = normalise_repair(repair)

        return {
            "is_error": True,
            "error_type": error_type,
            "evidence": raw_evidence,
            "repair": repair,
            "valid_json": True,
        }
