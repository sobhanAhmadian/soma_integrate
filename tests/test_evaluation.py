import numpy as np
import pytest
from src.soma_integ import evaluate_binary_classification, Result
import math


def test_evaluate_binary_classification():
    y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_predict = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.7])
    threshold = 0.5
    result = evaluate_binary_classification(y_test, y_predict, threshold)

    assert result.binary == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    assert result.acc == 0.7
    assert result.precision == 0.75
    assert result.recall == 0.6
    assert round(result.f1, 2) == 0.67