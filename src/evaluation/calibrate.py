#!/usr/bin/env python3
"""Confidence calibration and prior-shift correction for ContextGuard.

Author: Sahilpreet

Implements Milestone 3 decisions D14, D15, and M2 Section 14.3:
1. Temperature Scaling:
   Fits a scalar parameter T > 0 on the balanced validation split by minimizing
   binary cross-entropy:
       z = logit(p_error)
       p_calibrated = sigmoid(z / T)

2. Natural-Prior Correction:
   Adjusts temperature-scaled probabilities for natural table error prevalence:
       logit(p_natural) = (z / T) + log(pi / (1 - pi)) - log(p0 / (1 - p0))
       p_natural = sigmoid(logit(p_natural))

   where:
   - T: fitted scalar temperature
   - pi: natural error prevalence of the target table (0 < pi < 1)
   - p0: balanced training/validation prior (default 0.5, 0 < p0 < 1)

Fixed Integration Contract:
- fit_temperature(probs, labels, ...) -> (T, metadata)
- apply_temperature(probs, temperature) -> list of calibrated probabilities
- adjust_prior(probs, natural_prior, training_prior=0.5) -> list of adjusted probabilities
- calibrate_and_adjust(probs, temperature, natural_prior, training_prior=0.5) -> list
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EPSILON = 1e-12
LOGIT_BOUND = 50.0  # Prevents overflow in exp(-x) or exp(x)
MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 20.0


def logit(p: float, eps: float = EPSILON) -> float:
    """Compute numerically stable logit with probability clipping."""
    p_clipped = max(eps, min(1.0 - eps, p))
    return math.log(p_clipped) - math.log(1.0 - p_clipped)


def sigmoid(z: float, bound: float = LOGIT_BOUND) -> float:
    """Compute numerically stable sigmoid avoiding overflow."""
    z_bounded = max(-bound, min(bound, z))
    if z_bounded >= 0.0:
        return 1.0 / (1.0 + math.exp(-z_bounded))
    else:
        exp_z = math.exp(z_bounded)
        return exp_z / (1.0 + exp_z)


def binary_cross_entropy(
    logits: Sequence[float],
    labels: Sequence[float],
    temperature: float,
) -> float:
    """Compute mean binary cross-entropy loss at scalar temperature T > 0.

    Numerically stable formula using log1p and softplus:
        loss_i = max(x_i, 0) - x_i * y_i + log(1 + exp(-|x_i|))
    where x_i = z_i / T.
    """
    n = len(logits)
    if n == 0:
        return 0.0

    total_loss = 0.0
    for z, y in zip(logits, labels):
        x = z / temperature
        # Numerically stable BCE with logits
        # log(1 + exp(x)) - x * y  if x < 0 else log(1 + exp(-x)) + x * (1 - y)
        abs_x = abs(x)
        term1 = max(x, 0.0) - (x * y)
        term2 = math.log1p(math.exp(-abs_x))
        total_loss += term1 + term2

    return total_loss / n


def _golden_section_search(
    f,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, int]:
    """1D bounded scalar minimization via golden-section search.

    Deterministic and requires zero third-party dependencies.
    """
    invphi = (math.sqrt(5.0) - 1.0) / 2.0  # 1 / phi ~= 0.618033988749895
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)
    iterations = 0

    while abs(b - a) > tol and iterations < max_iter:
        iterations += 1
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = f(d)

    optimal_t = (a + b) / 2.0
    return optimal_t, iterations


def fit_temperature(
    probs: Sequence[float],
    labels: Sequence[Union[bool, int, float]],
    t_min: float = MIN_TEMPERATURE,
    t_max: float = MAX_TEMPERATURE,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, Dict[str, Any]]:
    """Fit optimal temperature scalar T on the balanced validation set.

    Args:
        probs: Sequence of predicted probabilities P(is_error) in [0, 1].
        labels: Sequence of ground-truth binary labels (True/False or 1/0).
        t_min: Minimum allowed temperature (must be > 0).
        t_max: Maximum allowed temperature.
        tol: Optimization convergence tolerance.
        max_iter: Maximum optimization iterations.

    Returns:
        (T, metadata):
            T (float): Optimal scalar temperature (> 0).
            metadata (dict): Optimization diagnostics including loss_before,
                loss_after, iterations, n_samples, and fit_success.
    """
    if len(probs) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(probs)} probabilities but {len(labels)} labels."
        )

    n_samples = len(probs)
    if n_samples == 0:
        raise ValueError("Cannot fit temperature on an empty dataset.")

    if t_min <= 0.0 or t_max <= t_min:
        raise ValueError(f"Invalid temperature bounds: [{t_min}, {t_max}].")

    # Validate probabilities and extract logits
    clean_logits: List[float] = []
    clean_labels: List[float] = []

    for i, (p, y) in enumerate(zip(probs, labels)):
        if p is None or not isinstance(p, (int, float)):
            raise ValueError(f"Invalid probability at index {i}: {p!r}")
        if math.isnan(p) or math.isinf(p) or p < 0.0 or p > 1.0:
            raise ValueError(
                f"Probability out of range [0, 1] or non-finite at index {i}: {p}"
            )

        if isinstance(y, bool):
            y_float = 1.0 if y else 0.0
        elif isinstance(y, (int, float)):
            if y not in (0, 1, 0.0, 1.0):
                raise ValueError(f"Label at index {i} must be binary, got: {y}")
            y_float = float(y)
        else:
            raise ValueError(f"Invalid label type at index {i}: {type(y)}")

        clean_logits.append(logit(p))
        clean_labels.append(y_float)

    # Check for single-class edge case
    n_pos = sum(clean_labels)
    if n_pos == 0 or n_pos == n_samples:
        # Single-class labels cannot establish temperature scaling
        bce_identity = binary_cross_entropy(clean_logits, clean_labels, 1.0)
        return 1.0, {
            "temperature": 1.0,
            "loss_before": bce_identity,
            "loss_after": bce_identity,
            "n_samples": n_samples,
            "iterations": 0,
            "fit_success": False,
            "note": "Single-class validation labels; returned identity temperature T=1.0.",
        }

    # Loss function for scalar optimization
    loss_fn = lambda t: binary_cross_entropy(clean_logits, clean_labels, t)

    loss_before = loss_fn(1.0)
    optimal_t, iters = _golden_section_search(
        loss_fn, a=t_min, b=t_max, tol=tol, max_iter=max_iter
    )
    loss_after = loss_fn(optimal_t)

    metadata = {
        "temperature": optimal_t,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_reduction": loss_before - loss_after,
        "n_samples": n_samples,
        "iterations": iters,
        "fit_success": True,
    }

    return optimal_t, metadata


def apply_temperature(
    probs: Sequence[float],
    temperature: float,
) -> List[float]:
    """Calibrate probabilities using a fitted temperature scalar T > 0.

    Transformation: p_calibrated = sigmoid(logit(p) / T).
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got: {temperature}")

    calibrated: List[float] = []
    for i, p in enumerate(probs):
        if p is None or not isinstance(p, (int, float)):
            raise ValueError(f"Invalid probability at index {i}: {p!r}")
        if math.isnan(p) or math.isinf(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Probability out of range [0, 1] at index {i}: {p}")

        z = logit(p)
        z_scaled = z / temperature
        calibrated.append(sigmoid(z_scaled))

    return calibrated


def adjust_prior(
    probs: Sequence[float],
    natural_prior: float,
    training_prior: float = 0.5,
) -> List[float]:
    """Adjust probabilities for natural table prevalence (prior-shift correction).

    Formula:
        logit(p_natural) = logit(p) + log(pi / (1 - pi)) - log(p0 / (1 - p0))
        p_natural = sigmoid(logit(p_natural))

    Args:
        probs: Predicted (or temperature-scaled) probabilities in [0, 1].
        natural_prior: Natural error prevalence pi of the target table (0 < pi < 1).
        training_prior: Balanced training/validation prior p0 (0 < p0 < 1, default 0.5).

    Returns:
        List of prior-adjusted probabilities.
    """
    if not (0.0 < natural_prior < 1.0):
        raise ValueError(
            f"Natural prior pi must be strictly in (0, 1), got: {natural_prior}"
        )
    if not (0.0 < training_prior < 1.0):
        raise ValueError(
            f"Training prior p0 must be strictly in (0, 1), got: {training_prior}"
        )

    # Shift term in log-odds
    shift = math.log(natural_prior / (1.0 - natural_prior)) - math.log(
        training_prior / (1.0 - training_prior)
    )

    adjusted: List[float] = []
    for i, p in enumerate(probs):
        if p is None or not isinstance(p, (int, float)):
            raise ValueError(f"Invalid probability at index {i}: {p!r}")
        if math.isnan(p) or math.isinf(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Probability out of range [0, 1] at index {i}: {p}")

        z = logit(p)
        z_shifted = z + shift
        adjusted.append(sigmoid(z_shifted))

    return adjusted


def calibrate_and_adjust(
    probs: Sequence[float],
    temperature: float,
    natural_prior: float,
    training_prior: float = 0.5,
) -> List[float]:
    """Apply temperature scaling followed by natural-prior adjustment.

    Transformation:
        z = logit(p) / T
        logit(p_natural) = z + log(pi / (1 - pi)) - log(p0 / (1 - p0))
        p_natural = sigmoid(logit(p_natural))
    """
    calibrated = apply_temperature(probs, temperature)
    return adjust_prior(calibrated, natural_prior, training_prior=training_prior)


class TemperatureScaler:
    """Class encapsulation for temperature scaling and natural-prior calibration."""

    def __init__(
        self,
        temperature: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got: {temperature}")
        self.temperature = float(temperature)
        self.metadata = metadata or {}

    @classmethod
    def fit(
        cls,
        probs: Sequence[float],
        labels: Sequence[Union[bool, int, float]],
        t_min: float = MIN_TEMPERATURE,
        t_max: float = MAX_TEMPERATURE,
    ) -> TemperatureScaler:
        """Fit TemperatureScaler on balanced validation data."""
        t, meta = fit_temperature(probs, labels, t_min=t_min, t_max=t_max)
        return cls(temperature=t, metadata=meta)

    def transform(self, probs: Sequence[float]) -> List[float]:
        """Apply temperature scaling to a sequence of probabilities."""
        return apply_temperature(probs, self.temperature)

    def adjust_prior(
        self,
        probs: Sequence[float],
        natural_prior: float,
        training_prior: float = 0.5,
    ) -> List[float]:
        """Apply natural-prior shift to a sequence of probabilities."""
        return adjust_prior(probs, natural_prior, training_prior=training_prior)

    def calibrate(
        self,
        probs: Sequence[float],
        natural_prior: float,
        training_prior: float = 0.5,
    ) -> List[float]:
        """Apply temperature scaling followed by natural-prior shift."""
        return calibrate_and_adjust(
            probs, self.temperature, natural_prior, training_prior=training_prior
        )
