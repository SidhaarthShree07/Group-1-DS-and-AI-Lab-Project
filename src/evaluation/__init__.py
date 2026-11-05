"""Evaluation, parsing, metrics, and calibration modules for ContextGuard."""

from src.evaluation.calibrate import (
    TemperatureScaler,
    adjust_prior,
    apply_temperature,
    calibrate_and_adjust,
    fit_temperature,
)
from src.evaluation.metrics import (
    calculate_metrics,
    export_reporting_grid,
    generate_reporting_grid,
    score,
)
from src.evaluation.parse import normalise_repair, parse_prediction

__all__ = [
    "parse_prediction",
    "normalise_repair",
    "score",
    "calculate_metrics",
    "generate_reporting_grid",
    "export_reporting_grid",
    "fit_temperature",
    "apply_temperature",
    "adjust_prior",
    "calibrate_and_adjust",
    "TemperatureScaler",
]
