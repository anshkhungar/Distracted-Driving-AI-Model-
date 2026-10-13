import pytest

from distracted_driving.metrics import evaluate_predictions


def test_perfect_predictions():
    result = evaluate_predictions(["safe", "unsafe"], ["safe", "unsafe"])
    assert result["accuracy"] == 1.0


def test_partial_accuracy():
    result = evaluate_predictions(["safe", "unsafe"], ["safe", "safe"])
    assert result["accuracy"] == 0.5


def test_invalid_input():
    with pytest.raises(ValueError):
        evaluate_predictions(["safe"], [])
