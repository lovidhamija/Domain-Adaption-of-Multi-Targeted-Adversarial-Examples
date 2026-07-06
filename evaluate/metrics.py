

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)


class ClassificationMetrics:
    """
    Computes all classification metrics.
    """

    def __init__(self, num_classes):

        self.num_classes = num_classes

    #######################################################
    # Accuracy
    #######################################################

    def accuracy(self, y_true, y_pred):

        return accuracy_score(y_true, y_pred)

    #######################################################
    # Precision
    #######################################################

    def precision(self, y_true, y_pred):

        return {

            "macro":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "micro":
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

            "weighted":
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
        }

    #######################################################
    # Recall
    #######################################################

    def recall(self, y_true, y_pred):

        return {

            "macro":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "micro":
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

            "weighted":
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
        }

    #######################################################
    # F1 Score
    #######################################################

    def f1(self, y_true, y_pred):

        return {

            "macro":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "micro":
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

            "weighted":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
        }

    #######################################################
    # Balanced Accuracy
    #######################################################

    def balanced_accuracy(self, y_true, y_pred):

        return balanced_accuracy_score(
            y_true,
            y_pred
        )

    #######################################################
    # Matthews Correlation
    #######################################################

    def mcc(self, y_true, y_pred):

        return matthews_corrcoef(
            y_true,
            y_pred
        )

    #######################################################
    # Cohen Kappa
    #######################################################

    def kappa(self, y_true, y_pred):

        return cohen_kappa_score(
            y_true,
            y_pred
        )

    #######################################################
    # Confusion Matrix
    #######################################################

    def confusion(self, y_true, y_pred):

        return confusion_matrix(
            y_true,
            y_pred
        )

    #######################################################
    # Classification Report
    #######################################################

    def report(self, y_true, y_pred):

        return classification_report(
            y_true,
            y_pred,
            digits=4
        )

    #######################################################
    # ROC-AUC
    #######################################################

    def roc_auc(self, y_true, probabilities):

        try:

            if self.num_classes == 2:

                return roc_auc_score(
                    y_true,
                    probabilities[:, 1]
                )

            else:

                return roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="macro"
                )

        except Exception:

            return None

    #######################################################
    # Average Precision
    #######################################################

    def average_precision(self, y_true, probabilities):

        try:

            if self.num_classes == 2:

                return average_precision_score(
                    y_true,
                    probabilities[:, 1]
                )

            return None

        except Exception:

            return None

    #######################################################
    # Top-k Accuracy
    #######################################################

    def top_k_accuracy(
            self,
            logits,
            labels,
            k=5):

        with torch.no_grad():

            _, pred = logits.topk(
                k,
                dim=1
            )

            pred = pred.t()

            correct = pred.eq(
                labels.view(1, -1).expand_as(pred)
            )

            correct_total = correct.reshape(-1).float().sum()

            return correct_total.item() / labels.size(0)

    #######################################################
    # Overall Evaluation
    #######################################################

    def evaluate(
            self,
            y_true,
            y_pred,
            probabilities=None):

        results = {}

        results["Accuracy"] = self.accuracy(
            y_true,
            y_pred
        )

        results["Precision"] = self.precision(
            y_true,
            y_pred
        )

        results["Recall"] = self.recall(
            y_true,
            y_pred
        )

        results["F1"] = self.f1(
            y_true,
            y_pred
        )

        results["Balanced Accuracy"] = self.balanced_accuracy(
            y_true,
            y_pred
        )

        results["MCC"] = self.mcc(
            y_true,
            y_pred
        )

        results["Kappa"] = self.kappa(
            y_true,
            y_pred
        )

        results["Confusion Matrix"] = self.confusion(
            y_true,
            y_pred
        )

        results["Classification Report"] = self.report(
            y_true,
            y_pred
        )

        if probabilities is not None:

            results["ROC-AUC"] = self.roc_auc(
                y_true,
                probabilities
            )

            results["Average Precision"] = self.average_precision(
                y_true,
                probabilities
            )

        return results
