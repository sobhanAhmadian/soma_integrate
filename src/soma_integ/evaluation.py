import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
    auc,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt


class Result:
    """
    Represents the evaluation result of a model.

    Attributes:
        auc (float): The Area Under the Curve (AUC) value.
        acc (float): The accuracy value.
        f1 (float): The F1 score value.
        aupr (float): The Area Under the Precision-Recall Curve (AUPR) value.
        loss (float): The loss value.
        recall (float): The recall value.
        precision (float): The precision value.
        mcc (float): The Matthews Correlation Coefficient (MCC) value.
        max_f1 (float): The maximum F1 score value for different thresholds.
        tpr (float): The True Positive Rate (TPR) value.
        fpr (float): The False Positive Rate (FPR) value.
        fpr_list (list): The list of False Positive Rate values for different thresholds.
        tpr_list (list): The list of True Positive Rate values for different thresholds.
        auc_list (list): The list of AUC values for different thresholds.

    Methods:
        get_result(): Returns the evaluation result as a dictionary.
        add(result): Adds the values of another Result object to this Result object.
        divide(k): Divides the values of this Result object by a given number.
    """

    def __init__(self) -> None:
        self.auc = 0
        self.acc = 0
        self.f1 = 0
        self.aupr = 0
        self.loss = 0
        self.recall = 0
        self.precision = 0
        self.mcc = 0
        self.max_f1 = 0
        self.tpr = 0
        self.fpr = 0
        self.fpr_list = []
        self.tpr_list = []
        self.auc_list = []

    def get_result(self):
        """
        Returns the evaluation result as a dictionary.

        Returns:
            dict: A dictionary containing the evaluation metrics and their corresponding values.
        """
        return {
            "AUC": self.auc,
            "ACC": self.acc,
            "F1 Score": self.f1,
            "AUPR": self.aupr,
            "Loss": self.loss,
            "Recall": self.recall,
            "Precision": self.precision,
            "MCC": self.mcc,
            "Max F1": self.max_f1,
            "True Positive Rate": self.tpr,
            "False Positive Rate": self.fpr,
        }


class CrossValidationResult:
    """
    Represents the result of cross-validation for evaluating a model's performance.

    Attributes:
        result (Result): The aggregated result of all folds.
        fold_results (list): List of individual fold results.
        is_result_calculated (bool): Indicates whether the result has been calculated.

    Functions:
        add_fold_result(test_result): Adds the result of a fold to the list of fold results.
        calculate_cv_result(): Calculates the aggregated result of all folds.
        get_roc_curve(ax): Plots the receiver operating characteristic (ROC) curve.
    """

    def __init__(self):
        self.result = Result()
        self.fold_results = []
        self.is_result_calculated = False
        self.k = 0

    def add_fold_result(self, test_result):
        """
        Adds the result of a fold to the list of fold results.

        Args:
            test_result (Result): The result of a fold.
        """
        self.fold_results.append(test_result)
        self.k += 1

    def calculate_cv_result(self):
        """
        Calculates the aggregated result of all folds.
        """
        for test_result in self.fold_results:
            self._accumulate(test_result)
        self._divide(self.k)
        self.is_result_calculated = True

    def _divide(self, k):
        """
        Divides the aggregated result by the number of folds.

        Args:
            k (int): The number of folds.
        """
        self.result.acc = self.result.acc / k
        self.result.f1 = self.result.f1 / k
        self.result.auc = self.result.auc / k
        self.result.aupr = self.result.aupr / k
        self.result.loss = self.result.loss / k
        self.result.recall = self.result.recall / k
        self.result.precision = self.result.precision / k
        self.result.mcc = self.result.mcc / k
        self.result.max_f1 = self.result.max_f1 / k
        self.result.tpr = self.result.tpr / k
        self.result.fpr = self.result.fpr / k
        self.result.fpr_list = (np.array(self.result.fpr_list) / k).tolist()
        self.result.tpr_list = (np.array(self.result.tpr_list) / k).tolist()
        self.result.auc_list = (np.array(self.result.auc_list) / k).tolist()

    def _accumulate(self, test_result):
        """
        Accumulates the result of a fold to the aggregated result.

        Args:
            test_result (Result): The result of a fold.
        """
        self.result.acc += test_result.acc
        self.result.f1 += test_result.f1
        self.result.auc += test_result.auc
        self.result.aupr += test_result.aupr
        self.result.loss += test_result.loss
        self.result.recall += test_result.recall
        self.result.precision += test_result.precision
        self.result.mcc += test_result.mcc
        self.result.max_f1 += test_result.max_f1
        self.result.tpr += test_result.tpr
        self.result.fpr += test_result.fpr
        self.result.fpr_list = (
            np.array(self.result.fpr_list) + np.array(test_result.fpr_list)
        ).tolist()
        self.result.tpr_list = (
            np.array(self.result.tpr_list) + np.array(test_result.tpr_list)
        ).tolist()
        self.result.auc_list = (
            np.array(self.result.auc_list) + np.array(test_result.auc_list)
        ).tolist()

    def get_roc_curve(self, ax):
        """
        Plots the receiver operating characteristic (ROC) curve.

        Args:
            ax: The matplotlib axes object to plot on.
        """
        if not self.is_result_calculated:
            self.calculate_cv_result()

        tprs = [result.tpr_list for result in self.fold_results]
        aucs = [result.auc_list for result in self.fold_results]
        mean_fpr = self.result.fpr_list

        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random", alpha=0.8
        )  # Plotting the random line

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )  # Plotting the mean ROC curve

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.3,
            label=r"$\pm$ 1 std. dev.",
        )  # Plotting the standard deviation

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic",
        )
        ax.legend(loc="lower right")


def evaluate_binary_classification(
    y_test, y_predict, threshold, mean_fpr_list=np.linspace(0, 1, 100)
):
    """
    Calculate various evaluation metrics for binary classification predictions.

    Args:
        y_test (array-like): True labels of the test set.
        y_predict (array-like): Predicted labels of the test set.
        threshold (float): Threshold value for converting predicted probabilities to binary predictions.
        mean_fpr_list (array-like, optional): List of mean false positive rates for ROC curve interpolation. 
            Defaults to np.linspace(0, 1, 100).

    Returns:
        Result: An object containing the calculated evaluation metrics.

    """

    result = Result()

    binary_predict = np.where(np.array(y_predict) >= threshold, 1, 0).tolist()

    result.acc = accuracy_score(y_test, binary_predict)
    result.f1 = f1_score(y_test, binary_predict, average="macro", zero_division=1)
    result.recall = recall_score(
        y_test, binary_predict, average="macro", zero_division=1
    )
    result.precision = precision_score(
        y_test, binary_predict, average="macro", zero_division=1
    )
    result.mcc = matthews_corrcoef(y_test, binary_predict)

    # Calculate max F1 score
    precision, recall, _ = precision_recall_curve(y_test, y_predict)
    numerator = 2 * recall * precision
    denominator = recall + precision
    f1_scores = np.divide(
        numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0)
    )
    result.max_f1 = np.max(f1_scores)

    # Calculate AUC, TPR, FPR
    fpr, tpr, _ = roc_curve(y_test, y_predict, pos_label=1)
    result.auc = auc(fpr, tpr)
    result.tpr = tpr
    result.fpr = fpr

    # Calculate FPR, TPR and AUC Lists
    viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=result.auc)
    interp_tpr = np.interp(mean_fpr_list, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0

    result.mean_fpr_list = [mean_fpr_list]
    result.tpr_list = [interp_tpr]
    result.auc_list = [viz.roc_auc]

    return result
