

import os
import csv
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import accuracy_score

from evaluation.metrics import ClassificationMetrics


class ClassifierEvaluator:

    """
    Evaluates the Shared Encoder + Class Predictor
    """

    def __init__(self,
                 encoder,
                 classifier,
                 device,
                 num_classes=100):

        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)

        self.device = device

        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.metrics = ClassificationMetrics(
            num_classes=num_classes
        )

    ############################################################
    # Evaluate
    ############################################################

    @torch.no_grad()
    def evaluate(
            self,
            dataloader):

        self.encoder.eval()
        self.classifier.eval()

        total_loss = 0

        total_correct = 0

        total_samples = 0

        ########################################################
        # Storage
        ########################################################

        feature_list = []

        logits_list = []

        probability_list = []

        prediction_list = []

        label_list = []

        ########################################################
        # Iterate
        ########################################################

        for images, labels in tqdm(
                dataloader,
                desc="Evaluating"):

            images = images.to(self.device)

            labels = labels.to(self.device)

            ##############################################
            # Forward
            ##############################################

            features = self.encoder(images)

            logits = self.classifier(features)

            ##############################################
            # Loss
            ##############################################

            loss = self.criterion(
                logits,
                labels
            )

            total_loss += loss.item() * labels.size(0)

            ##############################################
            # Prediction
            ##############################################

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            ##############################################
            # Accuracy
            ##############################################

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            ##############################################
            # Store everything
            ##############################################

            feature_list.append(

                features.detach().cpu()
            )

            logits_list.append(

                logits.detach().cpu()
            )

            probability_list.append(

                probabilities.detach().cpu()
            )

            prediction_list.append(

                predictions.detach().cpu()
            )

            label_list.append(

                labels.detach().cpu()
            )

        ########################################################
        # Concatenate
        ########################################################

        features = torch.cat(
            feature_list,
            dim=0
        )

        logits = torch.cat(
            logits_list,
            dim=0
        )

        probabilities = torch.cat(
            probability_list,
            dim=0
        )

        predictions = torch.cat(
            prediction_list,
            dim=0
        )

        labels = torch.cat(
            label_list,
            dim=0
        )

        ########################################################
        # Basic Results
        ########################################################

        avg_loss = total_loss / total_samples

        accuracy = (
            total_correct /
            total_samples
        )

        ########################################################
        # Top-5 Accuracy
        ########################################################

        top5_accuracy = self.metrics.top_k_accuracy(

            logits,

            labels,

            k=5
        )

        ########################################################
        # Convert to numpy
        ########################################################

        labels_np = labels.numpy()

        predictions_np = predictions.numpy()

        probabilities_np = probabilities.numpy()
                ########################################################
        # Compute Classification Metrics
        ########################################################

        results = self.metrics.evaluate(

            y_true=labels_np,

            y_pred=predictions_np,

            probabilities=probabilities_np

        )

        ########################################################
        # Add Extra Metrics
        ########################################################

        results["Loss"] = avg_loss

        results["Top1 Accuracy"] = accuracy

        results["Top5 Accuracy"] = top5_accuracy

        ########################################################
        # Return Everything
        ########################################################

        evaluation = {

            "metrics": results,

            "features": features.numpy(),

            "logits": logits.numpy(),

            "probabilities": probabilities_np,

            "predictions": predictions_np,

            "labels": labels_np

        }

        return evaluation

    ############################################################
    # Print Results
    ############################################################

    def print_results(self, evaluation):

        metrics = evaluation["metrics"]

        print("\n" + "=" * 70)

        print("CLASSIFICATION RESULTS")

        print("=" * 70)

        print(f"Loss               : {metrics['Loss']:.4f}")

        print(f"Top-1 Accuracy     : {metrics['Top1 Accuracy']*100:.2f}%")

        print(f"Top-5 Accuracy     : {metrics['Top5 Accuracy']*100:.2f}%")

        print(f"Balanced Accuracy  : {metrics['Balanced Accuracy']:.4f}")

        print(f"MCC                : {metrics['MCC']:.4f}")

        print(f"Cohen Kappa        : {metrics['Kappa']:.4f}")

        if metrics["ROC-AUC"] is not None:

            print(f"ROC-AUC            : {metrics['ROC-AUC']:.4f}")

        if metrics["Average Precision"] is not None:

            print(f"Average Precision  : {metrics['Average Precision']:.4f}")

        print("\nPrecision")

        print(metrics["Precision"])

        print("\nRecall")

        print(metrics["Recall"])

        print("\nF1")

        print(metrics["F1"])

        print("\nConfusion Matrix")

        print(metrics["Confusion Matrix"])

        print("\nClassification Report")

        print(metrics["Classification Report"])

        print("=" * 70)

    ############################################################
    # Save Metrics
    ############################################################

    def save_metrics(

            self,

            evaluation,

            save_dir,

            filename="classification_metrics.csv"):

        os.makedirs(

            save_dir,

            exist_ok=True

        )

        path = os.path.join(

            save_dir,

            filename

        )

        metrics = evaluation["metrics"]

        with open(

                path,

                "w",

                newline="") as file:

            writer = csv.writer(file)

            writer.writerow(["Metric", "Value"])

            writer.writerow(["Loss", metrics["Loss"]])

            writer.writerow(["Top1 Accuracy", metrics["Top1 Accuracy"]])

            writer.writerow(["Top5 Accuracy", metrics["Top5 Accuracy"]])

            writer.writerow(["Balanced Accuracy", metrics["Balanced Accuracy"]])

            writer.writerow(["MCC", metrics["MCC"]])

            writer.writerow(["Kappa", metrics["Kappa"]])

            writer.writerow(["ROC-AUC", metrics["ROC-AUC"]])

            writer.writerow(["Average Precision", metrics["Average Precision"]])

        print(f"\nMetrics saved to {path}")

    ############################################################
    # Save Predictions
    ############################################################

    def save_predictions(

            self,

            evaluation,

            save_dir,

            filename="predictions.csv"):

        os.makedirs(

            save_dir,

            exist_ok=True

        )

        path = os.path.join(

            save_dir,

            filename

        )

        labels = evaluation["labels"]

        predictions = evaluation["predictions"]

        with open(

                path,

                "w",

                newline="") as file:

            writer = csv.writer(file)

            writer.writerow(

                [

                    "GroundTruth",

                    "Prediction"

                ]

            )

            for gt, pred in zip(

                    labels,

                    predictions):

                writer.writerow(

                    [

                        int(gt),

                        int(pred)

                    ]

                )

        print(f"Predictions saved to {path}")

    ############################################################
    # Save Features
    ############################################################

    def save_features(

            self,

            evaluation,

            save_dir,

            filename="features.npy"):

        os.makedirs(

            save_dir,

            exist_ok=True

        )

        path = os.path.join(

            save_dir,

            filename

        )

        np.save(

            path,

            evaluation["features"]

        )

        print(f"Features saved to {path}")

    ############################################################
    # Save Probabilities
    ############################################################

    def save_probabilities(

            self,

            evaluation,

            save_dir,

            filename="probabilities.npy"):

        os.makedirs(

            save_dir,

            exist_ok=True

        )

        path = os.path.join(

            save_dir,

            filename

        )

        np.save(

            path,

            evaluation["probabilities"]

        )

        print(f"Probabilities saved to {path}")
        
