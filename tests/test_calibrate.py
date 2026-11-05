"""Unit tests for the calibration module (src/evaluation/calibrate.py).

Author: Sahilpreet

Tests verify all 14 required Milestone 3 calibration behaviors:
1. Temperature fitting on a balanced validation set
2. Positive temperature constraint
3. Temperature scaling transformation
4. Prior adjustment with pi == p0
5. Prior adjustment when pi < p0
6. Prior adjustment when pi > p0
7. Probability clipping and numerical stability
8. Invalid probabilities
9. Invalid priors
10. Empty input
11. Single-class validation labels
12. Reproducibility
13. Output shape preservation
14. Calibration parameter metadata and TemperatureScaler class
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Gracefully mock optional heavy dependencies if run in an environment without them
for _mod in ("numpy", "pandas"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from src.evaluation.calibrate import (
    TemperatureScaler,
    adjust_prior,
    apply_temperature,
    binary_cross_entropy,
    calibrate_and_adjust,
    fit_temperature,
    logit,
    sigmoid,
)
from src.evaluation.metrics import calculate_ece


class TestCalibrationModule(unittest.TestCase):
    """Test suite for temperature scaling and natural-prior adjustment."""

    def test_01_temperature_fitting_balanced_validation(self):
        """Test case 1: Fit temperature on an over-confident validation set.

        Overconfident predictions (e.g. 0.95 for True, 0.95 for False)
        should result in T > 1.0, softening logits and lowering BCE loss.
        """
        # Balanced 4 samples: 2 True, 2 False
        # Over-confident scores
        probs = [0.95, 0.95, 0.05, 0.05]
        labels = [True, False, True, False]

        t, meta = fit_temperature(probs, labels)
        self.assertGreater(t, 1.0)
        self.assertTrue(meta["fit_success"])
        self.assertLess(meta["loss_after"], meta["loss_before"])
        self.assertEqual(meta["n_samples"], 4)

    def test_02_positive_temperature_constraint(self):
        """Test case 2: Temperature must be strictly positive."""
        probs = [0.8, 0.2]
        labels = [True, False]
        t, _ = fit_temperature(probs, labels)
        self.assertGreater(t, 0.0)

        # Negative or zero temperature in apply_temperature must raise ValueError
        with self.assertRaises(ValueError):
            apply_temperature([0.5], temperature=0.0)
        with self.assertRaises(ValueError):
            apply_temperature([0.5], temperature=-1.5)
        with self.assertRaises(ValueError):
            TemperatureScaler(temperature=-0.5)

    def test_03_temperature_scaling_transformation(self):
        """Test case 3: Mathematical exactness of p_calibrated = sigmoid(logit(p) / T)."""
        p = 0.8
        t = 2.0
        # logit(0.8) = ln(0.8 / 0.2) = ln(4) ~= 1.38629436
        # z / T = ln(4) / 2 = ln(2) ~= 0.69314718
        # sigmoid(ln(2)) = 2 / (1 + 2) = 2/3 ~= 0.66666667
        calibrated = apply_temperature([p], temperature=t)
        self.assertAlmostEqual(calibrated[0], 2.0 / 3.0, places=6)

        # When T = 1.0, probabilities are unchanged
        identity_p = apply_temperature([p, 0.2, 0.5], temperature=1.0)
        self.assertAlmostEqual(identity_p[0], 0.8, places=6)
        self.assertAlmostEqual(identity_p[1], 0.2, places=6)
        self.assertAlmostEqual(identity_p[2], 0.5, places=6)

    def test_04_prior_adjustment_neutral_prior(self):
        """Test case 4: Prior adjustment when pi == p0 leaves probabilities unchanged."""
        probs = [0.2, 0.5, 0.85]
        # When target natural prior pi equals training prior p0:
        adjusted = adjust_prior(probs, natural_prior=0.5, training_prior=0.5)
        for orig, adj in zip(probs, adjusted):
            self.assertAlmostEqual(orig, adj, places=6)

        adjusted_custom = adjust_prior(probs, natural_prior=0.3, training_prior=0.3)
        for orig, adj in zip(probs, adjusted_custom):
            self.assertAlmostEqual(orig, adj, places=6)

    def test_05_prior_adjustment_lower_prevalence(self):
        """Test case 5: Prior adjustment when pi < p0 reduces probabilities (rare error setting)."""
        probs = [0.1, 0.5, 0.9]
        # Natural error rate 2.5% (Hospital) vs training prior 50%
        adjusted = adjust_prior(probs, natural_prior=0.025, training_prior=0.5)
        for orig, adj in zip(probs, adjusted):
            self.assertLess(adj, orig)

        # For p = 0.5 (logit 0):
        # logit(adj) = 0 + ln(0.025 / 0.975) - 0 = ln(1 / 39)
        # adj = (1/39) / (1 + 1/39) = 1/40 = 0.025
        self.assertAlmostEqual(adjusted[1], 0.025, places=6)

    def test_06_prior_adjustment_higher_prevalence(self):
        """Test case 6: Prior adjustment when pi > p0 increases probabilities."""
        probs = [0.1, 0.5, 0.8]
        # Natural error rate 80% vs training prior 50%
        adjusted = adjust_prior(probs, natural_prior=0.8, training_prior=0.5)
        for orig, adj in zip(probs, adjusted):
            self.assertGreater(adj, orig)

        # For p = 0.5:
        # logit(adj) = 0 + ln(0.8 / 0.2) = ln(4)
        # adj = 4 / (1 + 4) = 0.8
        self.assertAlmostEqual(adjusted[1], 0.8, places=6)

    def test_07_probability_clipping_and_numerical_stability(self):
        """Test case 7: Numerical stability for extreme probabilities and logits."""
        extreme_probs = [0.0, 1.0, 1e-15, 1.0 - 1e-15]
        # Must not raise ZeroDivisionError or OverflowError
        calibrated = apply_temperature(extreme_probs, temperature=1.5)
        self.assertEqual(len(calibrated), 4)
        for p in calibrated:
            self.assertFalse(math.isnan(p))
            self.assertFalse(math.isinf(p))
            self.assertTrue(0.0 <= p <= 1.0)

        # Extreme logits in sigmoid
        self.assertAlmostEqual(sigmoid(100.0), 1.0)
        self.assertAlmostEqual(sigmoid(-100.0), 0.0)

    def test_08_invalid_probabilities_raise_error(self):
        """Test case 8: Invalid probabilities (<0, >1, NaN, inf, None) raise ValueError."""
        invalid_cases = [-0.1, 1.2, float("nan"), float("inf"), None, "0.5"]
        for bad_p in invalid_cases:
            with self.subTest(bad_p=bad_p):
                with self.assertRaises(ValueError):
                    fit_temperature([bad_p], [True])
                with self.assertRaises(ValueError):
                    apply_temperature([bad_p], temperature=1.0)
                with self.assertRaises(ValueError):
                    adjust_prior([bad_p], natural_prior=0.1)

    def test_08b_invalid_labels_raise_error(self):
        """Test case 8b: Labels other than 0 and 1 (or booleans) must raise ValueError."""
        invalid_labels = [2, -1, 0.5, "1", "True", None, [1]]
        for bad_y in invalid_labels:
            with self.subTest(bad_y=bad_y):
                with self.assertRaises(ValueError):
                    fit_temperature([0.5], [bad_y])

    def test_09_invalid_priors_raise_error(self):
        """Test case 9: Invalid priors (<= 0 or >= 1) raise ValueError."""
        with self.assertRaises(ValueError):
            adjust_prior([0.5], natural_prior=0.0)
        with self.assertRaises(ValueError):
            adjust_prior([0.5], natural_prior=1.0)
        with self.assertRaises(ValueError):
            adjust_prior([0.5], natural_prior=-0.2)
        with self.assertRaises(ValueError):
            adjust_prior([0.5], natural_prior=0.2, training_prior=0.0)
        with self.assertRaises(ValueError):
            adjust_prior([0.5], natural_prior=0.2, training_prior=1.0)

    def test_10_empty_input(self):
        """Test case 10: Empty input handling."""
        with self.assertRaises(ValueError):
            fit_temperature([], [])
        self.assertEqual(apply_temperature([], temperature=1.0), [])
        self.assertEqual(adjust_prior([], natural_prior=0.1), [])
        self.assertEqual(calibrate_and_adjust([], temperature=1.0, natural_prior=0.1), [])

    def test_11_single_class_validation_labels(self):
        """Test case 11: Single-class validation data safely handled without crashing."""
        # All positive
        t1, meta1 = fit_temperature([0.7, 0.9], [True, True])
        self.assertEqual(t1, 1.0)
        self.assertFalse(meta1["fit_success"])
        self.assertIn("Single-class", meta1["note"])

        # All negative
        t2, meta2 = fit_temperature([0.1, 0.3], [False, False])
        self.assertEqual(t2, 1.0)
        self.assertFalse(meta2["fit_success"])

    def test_12_reproducibility(self):
        """Test case 12: Fitting procedure is fully deterministic."""
        probs = [0.9, 0.8, 0.3, 0.1, 0.65, 0.4]
        labels = [True, True, False, False, True, False]

        t1, meta1 = fit_temperature(probs, labels)
        t2, meta2 = fit_temperature(probs, labels)
        self.assertEqual(t1, t2)
        self.assertEqual(meta1["loss_after"], meta2["loss_after"])
        self.assertEqual(meta1["iterations"], meta2["iterations"])

    def test_13_output_shape_preservation(self):
        """Test case 13: Input length and order are strictly preserved."""
        probs = [0.12, 0.98, 0.45, 0.67, 0.33]
        out_temp = apply_temperature(probs, temperature=1.2)
        out_prior = adjust_prior(probs, natural_prior=0.25)
        out_both = calibrate_and_adjust(probs, temperature=1.2, natural_prior=0.25)

        self.assertEqual(len(out_temp), len(probs))
        self.assertEqual(len(out_prior), len(probs))
        self.assertEqual(len(out_both), len(probs))

    def test_14_temperature_scaler_class_api(self):
        """Test case 14: TemperatureScaler class API."""
        probs = [0.9, 0.8, 0.2, 0.1]
        labels = [True, True, False, False]

        scaler = TemperatureScaler.fit(probs, labels)
        self.assertGreater(scaler.temperature, 0.0)
        self.assertIn("loss_before", scaler.metadata)

        calibrated = scaler.transform(probs)
        self.assertEqual(len(calibrated), len(probs))

        adjusted = scaler.adjust_prior(probs, natural_prior=0.1)
        self.assertEqual(len(adjusted), len(probs))

        both = scaler.calibrate(probs, natural_prior=0.1)
        self.assertEqual(len(both), len(probs))

        # Assert functional API and class API produce identical results
        self.assertEqual(calibrated, apply_temperature(probs, scaler.temperature))
        self.assertEqual(adjusted, adjust_prior(probs, natural_prior=0.1))
        self.assertEqual(both, calibrate_and_adjust(probs, scaler.temperature, natural_prior=0.1))

        # Check metadata fields explicitly
        meta = scaler.metadata
        self.assertIn("temperature", meta)
        self.assertIn("loss_before", meta)
        self.assertIn("loss_after", meta)
        self.assertIn("n_samples", meta)
        self.assertIn("fit_success", meta)
        self.assertTrue(meta["fit_success"])

    def test_15_ece_perfectly_calibrated_synthetic_example(self):
        """Test case 15: M3 Plan Requirement - ECE of a perfectly calibrated toy example is close to 0.

        Creates synthetic bins where the mean confidence strictly equals the empirical error rate.
        Independent of optimization loss, the resulting ECE must be mathematically 0.0.
        """
        probs = []
        labels = []

        # 5 confidence bins across [0, 1]
        # Bin 1 (conf 0.1): 1 True, 9 False -> mean_conf = 0.1, acc = 0.1
        probs.extend([0.1] * 10)
        labels.extend([True] * 1 + [False] * 9)

        # Bin 2 (conf 0.3): 3 True, 7 False -> mean_conf = 0.3, acc = 0.3
        probs.extend([0.3] * 10)
        labels.extend([True] * 3 + [False] * 7)

        # Bin 3 (conf 0.5): 5 True, 5 False -> mean_conf = 0.5, acc = 0.5
        probs.extend([0.5] * 10)
        labels.extend([True] * 5 + [False] * 5)

        # Bin 4 (conf 0.7): 7 True, 3 False -> mean_conf = 0.7, acc = 0.7
        probs.extend([0.7] * 10)
        labels.extend([True] * 7 + [False] * 3)

        # Bin 5 (conf 0.9): 9 True, 1 False -> mean_conf = 0.9, acc = 0.9
        probs.extend([0.9] * 10)
        labels.extend([True] * 9 + [False] * 1)

        ece = calculate_ece(labels, probs, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
