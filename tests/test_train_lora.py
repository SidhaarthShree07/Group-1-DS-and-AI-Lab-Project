"""Early stopping, step count and learning-rate schedule of src/llm/train_lora.py (no model needed)."""
import math

import pytest

from src.llm.train_lora import EarlyStopper, lr_multiplier, total_steps


def test_early_stopper_stops_after_patience():
    s = EarlyStopper(patience=3)
    losses = [1.0, 0.8, 0.7, 0.71, 0.72, 0.70, 0.75]  # best 0.7 at step 2; three evaluations without improvement
    stops = []
    for i, v in enumerate(losses):
        s.update(v, i)
        stops.append(s.should_stop)
    assert s.best == 0.7 and s.best_step == 2
    assert stops == [False, False, False, False, False, True, True]


def test_early_stopper_off_and_reset():
    s = EarlyStopper(patience=0)
    for i, v in enumerate([1.0, 2.0, 3.0, 4.0]):
        s.update(v, i)
    assert not s.should_stop
    s = EarlyStopper(patience=2)
    for i, v in enumerate([1.0, 1.1, 0.9, 1.0]):  # the improvement at step 2 resets the count
        s.update(v, i)
    assert not s.should_stop and s.best_step == 2


def test_min_delta():
    s = EarlyStopper(patience=1, min_delta=0.01)
    s.update(1.0, 0)
    assert not s.update(0.995, 1)  # smaller than min_delta: not an improvement
    assert s.should_stop


def test_total_steps():
    assert total_steps(100, 0, 55091, 32) == 100
    assert total_steps(0, 1, 55091, 32) == 1722
    assert total_steps(100, 1, 55091, 32) == 1722  # --epochs overrides the default --max-steps
    assert total_steps(0, 0.5, 64, 32) == 1
    with pytest.raises(ValueError):
        total_steps(0, 0, 100, 32)


def test_lr_schedule():
    assert lr_multiplier(0, 10, 100) == pytest.approx(0.1)
    assert lr_multiplier(10, 10, 100) == pytest.approx(1.0)
    assert lr_multiplier(100, 10, 100) == pytest.approx(0.0, abs=1e-12)
    assert lr_multiplier(55, 10, 100) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))
    assert lr_multiplier(80, 10, 100, "constant") == 1.0
