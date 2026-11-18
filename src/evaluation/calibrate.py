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
- estimate_prior_em(probs, training_prior=0.5) -> (rate, info)
- fit_gate(probs, is_error, repair_ok, max_false_repair=0.10) -> {"flag": t1, "repair": t2}
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


# Fallback prior: computed from real candidates on the final validation run.
# Updated from the real m4_e6_d7_ft run (median of per-table candidate error rates:
#   beers ~0.266, flights ~0.860, hospital ~0.141 → median ≈ 0.266).
# This constant is kept for backward compatibility; fit_calibration.py computes it live.
# Thresholds that trigger the EM fallback (Fix 5 / Leader Update)
_EM_CLAMP_EPS: float = 1e-4  # estimate at or within eps of boundary (e.g. >= 0.9999 or <= 0.0001)


def estimate_prior_em(
    probs: Sequence[float],
    training_prior: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 100,
    fallback_rate: Optional[float] = None,
    max_deviation_from_default: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Estimate true class prevalence via EM prior-shift correction (Saerens et al., 2002).

    Given classifier probabilities P_0(y=1|x) obtained under balanced training/validation
    prior p0 (default 0.5), iteratively finds the maximum-likelihood target prior pi
    without requiring ground-truth labels for the target table:
        1. E-step: Compute adjusted posteriors P(y=1 | x_i, pi^(k)) via Bayes' rule.
        2. M-step: Set pi^(k+1) = (1 / N) * sum_i P(y=1 | x_i, pi^(k)).
        3. Repeat until |pi^(k+1) - pi^(k)| < tol or max_iter is reached.

    Fallback to ``fallback_rate`` (or sample average if None) is applied when EM fails:
      - The estimate is non-finite (NaN / Inf).
      - EM does not converge within ``max_iter`` iterations.
      - The final estimate is at the clamped boundary (within ``_EM_CLAMP_EPS``).
      - If ``max_deviation_from_default`` is provided and ``fallback_rate`` is set,
        when |curr_pi - fallback_rate| > max_deviation_from_default.

    The raw EM estimate prior to fallback is recorded in ``info["raw_em_estimate"]``.
    When the fallback fires, ``info["used_fallback"]`` is True and
    ``info["fallback_reason"]`` contains a human-readable explanation.

    Args:
        probs: Sequence of predicted probabilities in [0, 1].
        training_prior: Balanced training prior p0 in (0, 1), default 0.5.
        tol: Convergence tolerance for absolute prior change.
        max_iter: Maximum EM iterations.
        fallback_rate: Optional fallback prior when EM fails. If None, defaults to initial sample mean.
        max_deviation_from_default: Optional maximum deviation from fallback_rate to trigger fallback.

    Returns:
        (rate, info):
            rate (float): Estimated target prior in (0, 1).
            info (dict): Diagnostic metadata including iterations, convergence status,
                raw_em_estimate, used_fallback, fallback_reason, and initial/history estimates.
    """
    if len(probs) == 0:
        raise ValueError("Cannot estimate prior from an empty sequence of probabilities.")
    if not (0.0 < training_prior < 1.0):
        raise ValueError(f"training_prior must be strictly in (0, 1), got: {training_prior}")

    clean_probs: List[float] = []
    for i, p in enumerate(probs):
        if p is None or not isinstance(p, (int, float)):
            raise ValueError(f"Invalid probability at index {i}: {p!r}")
        if math.isnan(p) or math.isinf(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Probability out of range [0, 1] at index {i}: {p}")
        clean_probs.append(float(p))

    n = len(clean_probs)
    eps = 1e-6
    # Initialize from the sample average (clamped safely)
    init_pi = max(eps, min(1.0 - eps, sum(clean_probs) / n))
    curr_pi = init_pi
    history: List[float] = [curr_pi]
    converged = False
    iterations = 0

    p0 = training_prior

    for it in range(1, max_iter + 1):
        iterations = it
        # E-step: adjust posteriors with curr_pi
        c_pos = curr_pi / p0
        c_neg = (1.0 - curr_pi) / (1.0 - p0)

        sum_posteriors = 0.0
        for p in clean_probs:
            num = c_pos * p
            den = num + c_neg * (1.0 - p)
            if den <= 0.0:
                post = p
            else:
                post = num / den
            sum_posteriors += post

        new_pi = sum_posteriors / n
        new_pi = max(eps, min(1.0 - eps, new_pi))
        history.append(new_pi)

        if abs(new_pi - curr_pi) < tol:
            converged = True
            curr_pi = new_pi
            break
        curr_pi = new_pi

    raw_em_val = float(curr_pi)

    # Determine whether to fall back
    used_fallback = False
    fallback_reason = ""
    effective_fallback = fallback_rate if fallback_rate is not None else init_pi

    if math.isnan(curr_pi) or math.isinf(curr_pi):
        used_fallback = True
        fallback_reason = "EM estimate is non-finite (NaN/Inf)"
        curr_pi = effective_fallback
    elif not converged:
        used_fallback = True
        fallback_reason = f"EM did not converge within {max_iter} iterations"
        curr_pi = effective_fallback
    elif abs(curr_pi - eps) < _EM_CLAMP_EPS or abs(curr_pi - (1.0 - eps)) < _EM_CLAMP_EPS:
        used_fallback = True
        fallback_reason = "EM estimate hit the clamped boundary (eps or 1-eps)"
        curr_pi = effective_fallback
    elif max_deviation_from_default is not None and fallback_rate is not None:
        if abs(curr_pi - fallback_rate) > max_deviation_from_default:
            used_fallback = True
            fallback_reason = (
                f"EM estimate {curr_pi:.4f} deviates by "
                f"{abs(curr_pi - fallback_rate):.4f} > {max_deviation_from_default} "
                f"from the default rate {fallback_rate:.4f}"
            )
            curr_pi = effective_fallback

    info: Dict[str, Any] = {
        "rate": float(curr_pi),
        "raw_em_estimate": raw_em_val,
        "iterations": iterations,
        "converged": converged,
        "delta": abs(history[-1] - history[-2]) if len(history) > 1 else 0.0,
        "initial_rate": init_pi,
        "history": history,
        "training_prior": training_prior,
        "n_samples": n,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
    }
    return float(curr_pi), info


def fit_gate(
    probs: Sequence[float],
    is_error: Sequence[Union[bool, int, float]],
    repair_ok: Optional[Sequence[Optional[Union[bool, int, float]]]] = None,
    max_false_repair: float = 0.10,
    t_step: float = 0.005,
    repair_probs: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    """Fit optimal flag threshold (t1) and repair threshold (t2) on validation data.

    Decision Rule:
    - Flag threshold t1: Maximizes detection F1 on validation cells.
    - Repair threshold t2: Lowest confidence threshold whose empirical false-repair rate
      on validation is at most max_false_repair (default 10%), ensuring t2 >= t1.

    Args:
        probs: Sequence of error probabilities P(is_error) in [0, 1].
        is_error: Sequence of ground-truth binary labels (True for errors).
        repair_ok: Optional sequence indicating whether proposed repair is correct
            (True = correct repair, False = incorrect repair, None = no repair proposed).
        max_false_repair: Maximum tolerable false repair rate (default 0.10).
        t_step: Threshold discretization grid step.
        repair_probs: Optional sequence of repair-specific confidence scores (e.g. p_repair_min)
            to sweep for the repair threshold t2. If None, falls back to error probs.

    Returns:
        {"flag": t1, "repair": t2, "cap_met": cap_met}
    """
    if len(probs) != len(is_error):
        raise ValueError(f"Mismatch: {len(probs)} probs vs {len(is_error)} labels.")
    if repair_ok is not None and len(repair_ok) != len(probs):
        raise ValueError(f"Mismatch: {len(probs)} probs vs {len(repair_ok)} repair_ok.")
    if repair_probs is not None and len(repair_probs) != len(probs):
        raise ValueError(f"Mismatch: {len(probs)} probs vs {len(repair_probs)} repair_probs.")
    if len(probs) == 0:
        raise ValueError("Cannot fit gate thresholds on empty inputs.")

    clean_probs = [float(p) for p in probs]
    clean_labels = [bool(y) for y in is_error]
    has_repairs = repair_ok is not None

    # 1. Sweep to find t1 that maximizes detection F1
    best_f1 = -1.0
    best_t1 = 0.50
    best_prec_at_t1 = 0.0

    # Test thresholds from unique sample probabilities and uniform grid
    grid_steps = int(round(1.0 / t_step))
    uniform_grid = [round(i * t_step, 4) for i in range(1, grid_steps)]
    candidate_thresholds = sorted(list(set(uniform_grid + [round(p, 4) for p in clean_probs if 0.01 <= p <= 0.99])))

    for t in candidate_thresholds:
        tp = 0
        fp = 0
        fn = 0
        for p, y in zip(clean_probs, clean_labels):
            flagged = p >= t
            if flagged and y:
                tp += 1
            elif flagged and not y:
                fp += 1
            elif not flagged and y:
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Break ties by preferring standard 0.50 or higher precision
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and abs(t - 0.50) < abs(best_t1 - 0.50)):
            best_f1 = f1
            best_t1 = t
            best_prec_at_t1 = prec

    # 2. Sweep to find repair threshold t2 (lowest threshold with false-repair rate <= max_false_repair)
    best_t2 = max(best_t1, 0.80)
    cap_met: Optional[bool] = None  # None when there is no repair data
    if has_repairs:
        # Extract pairs (prob, is_repair_ok) where a repair was proposed
        repair_pairs = []
        for i, (p, ok) in enumerate(zip(clean_probs, repair_ok)):
            if ok is not None:
                r_p = repair_probs[i] if repair_probs is not None else None
                prob_val = float(r_p) if (r_p is not None and not math.isnan(r_p)) else p
                repair_pairs.append((prob_val, bool(ok)))

        if repair_pairs:
            qualifying_thresholds: List[float] = []
            min_frr = 1.0
            best_frr_t = 0.95

            # Search across observed candidate confidence levels and grid thresholds >= best_t1
            obs_confs = sorted(list(set(round(p, 4) for p, _ in repair_pairs if round(p, 4) >= best_t1)))
            t2_candidates = obs_confs if obs_confs else [t for t in candidate_thresholds if t >= best_t1]

            for t in t2_candidates:
                executed = [ok for p, ok in repair_pairs if p >= t]
                if not executed:
                    continue
                n_exec = len(executed)
                n_wrong = sum(1 for ok in executed if not ok)
                frr = n_wrong / n_exec
                if frr < min_frr:
                    min_frr = frr
                    best_frr_t = t
                if frr <= max_false_repair:
                    qualifying_thresholds.append(t)

            if qualifying_thresholds:
                # Lowest confidence meeting the false-repair cap
                best_t2 = min(qualifying_thresholds)
                cap_met = True
            else:
                best_t2 = best_frr_t
                cap_met = False  # max_false_repair constraint could not be satisfied

    return {"flag": round(best_t1, 4), "repair": round(best_t2, 4), "cap_met": cap_met}

