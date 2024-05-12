import numpy as np
import pytest
from src.soma_integ import evaluate_binary_classification, Result, CrossValidationResult
import math
import matplotlib.pyplot as plt


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


def test_cross_validation_result_add_fold_result():
    cv_result = CrossValidationResult()
    fold_result = Result()
    fold_result.acc = 0.8
    fold_result.f1 = 0.75
    fold_result.auc = 0.9

    cv_result.add_fold_result(fold_result)

    assert len(cv_result.fold_results) == 1
    assert cv_result.fold_results[0].acc == 0.8
    assert cv_result.fold_results[0].f1 == 0.75
    assert cv_result.fold_results[0].auc == 0.9


def test_cross_validation_result_calculate_cv_result():
    cv_result = CrossValidationResult()
    fold_result1 = Result()
    fold_result1.acc = 0.8
    fold_result1.f1 = 0.75
    fold_result1.auc = 0.9
    fold_result2 = Result()
    fold_result2.acc = 0.7
    fold_result2.f1 = 0.65
    fold_result2.auc = 0.85

    cv_result.add_fold_result(fold_result1)
    cv_result.add_fold_result(fold_result2)
    cv_result.calculate_cv_result()

    assert cv_result.result.acc == 0.75
    assert cv_result.result.f1 == 0.7
    assert cv_result.result.auc == 0.875


def test_cross_validation_result_get_roc_curve():
    cv_result = CrossValidationResult()
    fold_result1 = Result()
    fold_result1.fpr_list = [0.1, 0.2, 0.3]
    fold_result1.tpr_list = [0.4, 0.5, 0.6]
    fold_result1.auc_list = [0.7]
    fold_result2 = Result()
    fold_result2.fpr_list = [0.2, 0.3, 0.4]
    fold_result2.tpr_list = [0.5, 0.6, 0.7]
    fold_result2.auc_list = [0.8]

    cv_result.add_fold_result(fold_result1)
    cv_result.add_fold_result(fold_result2)

    ax = plt.subplot()
    cv_result.get_roc_curve(ax)
