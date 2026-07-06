

import os
import csv
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from evaluation.metrics import ClassificationMetrics


class DomainEvaluator:

    def __init__(
            self,
            encoder,
            discriminator,
            device,
            num_domains=5):

        self.encoder = encoder.to(device)

        self.discriminator = discriminator.to(device)

        self.device = device

        self.num_domains = num_domains

        self.criterion = nn.CrossEntropyLoss()

        self.metrics = ClassificationMetrics(
            num_classes=num_domains
        )

    ############################################################
    # Evaluate Domain Discriminator
    ############################################################

    @torch.no_grad()
    def evaluate(
            self,
            dataloader):

        self.encoder.eval()

        self.discriminator.eval()

        total_loss = 0.0

        total_correct = 0

        total_samples = 0

        ###############################################
        # Storage
        ###############################################

        feature_list = []

        logits_list = []

        probability_list = []

        prediction_list = []

        domain_label_list = []

        class_label_list = []

        ###############################################
        # Evaluation Loop
        ###############################################

        for images, class_labels, domain_labels in tqdm(

                dataloader,

                desc="Evaluating Domains"

        ):

            images = images.to(self.device)

            class_labels = class_labels.to(self.device)

            domain_labels = domain_labels.to(self.device)

            ###########################################
            # Shared Encoder
            ###########################################

            features = self.encoder(images)

            ###########################################
            # Domain Prediction
            ###########################################

            logits = self.discriminator(features)

            ###########################################
            # Loss
            ###########################################

            loss = self.criterion(

                logits,

                domain_labels

            )

            total_loss += (

                loss.item()

                * domain_labels.size(0)

            )

            ###########################################
            # Prediction
            ###########################################

            probabilities = torch.softmax(

                logits,

                dim=1

            )

            predictions = torch.argmax(

                probabilities,

                dim=1

            )

            ###########################################
            # Accuracy
            ###########################################

            total_correct += (

                predictions == domain_labels

            ).sum().item()

            total_samples += domain_labels.size(0)

            ###########################################
            # Store Everything
            ###########################################

            feature_list.append(

                features.cpu()

            )

            logits_list.append(

                logits.cpu()

            )

            probability_list.append(

                probabilities.cpu()

            )

            prediction_list.append(

                predictions.cpu()

            )

            domain_label_list.append(

                domain_labels.cpu()

            )

            class_label_list.append(

                class_labels.cpu()

            )

        ###################################################
        # Concatenate
        ###################################################

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

        domain_labels = torch.cat(

            domain_label_list,

            dim=0

        )

        class_labels = torch.cat(

            class_label_list,

            dim=0

        )

        ###################################################
        # Statistics
        ###################################################

        average_loss = (

            total_loss /

            total_samples

        )

        accuracy = (

            total_correct /

            total_samples

        )

        ###################################################
        # Convert to NumPy
        ###################################################

        labels_np = domain_labels.numpy()
                ############################################################
        # Compute Metrics
        ############################################################

        results = self.metrics.evaluate(

            y_true=labels_np,

            y_pred=predictions_np,

            probabilities=probabilities_np

        )

        ############################################################
        # Additional Statistics
        ############################################################

        results["Loss"] = average_loss

        results["Domain Accuracy"] = accuracy

        ############################################################
        # Per-Domain Accuracy
        ############################################################

        domain_names = {

            0: "Clean",

            1: "FGSM",

            2: "PGD",

            3: "CW",

            4: "Patch"

        }

        per_domain_accuracy = {}

        for domain_id, domain_name in domain_names.items():

            indices = labels_np == domain_id

            if np.sum(indices) == 0:

                per_domain_accuracy[domain_name] = 0.0

                continue

            correct = np.sum(

                predictions_np[indices] == labels_np[indices]

            )

            total = np.sum(indices)

            per_domain_accuracy[domain_name] = correct / total

        results["Per Domain Accuracy"] = per_domain_accuracy

        ############################################################
        # Return Everything
        ############################################################

        evaluation = {

            "metrics": results,

            "features": features.numpy(),

            "logits": logits.numpy(),

            "probabilities": probabilities_np,

            "predictions": predictions_np,

            "domain_labels": labels_np,

            "class_labels": class_labels.numpy()

        }

        return evaluation

    ############################################################
    # Pretty Printing
    ############################################################

    def print_results(

            self,

            evaluation):

        metrics = evaluation["metrics"]

        print("\n")

        print("=" * 70)

        print("DOMAIN DISCRIMINATOR EVALUATION")

        print("=" * 70)

        print(f"Loss               : {metrics['Loss']:.4f}")

        print(f"Domain Accuracy    : {metrics['Domain Accuracy']*100:.2f}%")

        print(f"Balanced Accuracy  : {metrics['Balanced Accuracy']:.4f}")

        print(f"MCC                : {metrics['MCC']:.4f}")

        print(f"Cohen Kappa        : {metrics['Kappa']:.4f}")

        if metrics["ROC-AUC"] is not None:

            print(f"ROC-AUC            : {metrics['ROC-AUC']:.4f}")

        if metrics["Average Precision"] is not None:

            print(f"Average Precision  : {metrics['Average Precision']:.4f}")

        print("\n")

        print("Per Domain Accuracy")

        print("-" * 40)

        for domain, value in metrics["Per Domain Accuracy"].items():

            print(f"{domain:<10}: {value*100:.2f}%")

        print("\n")

        print("Precision")

        print(metrics["Precision"])

        print("\n")

        print("Recall")

        print(metrics["Recall"])

        print("\n")

        print("F1 Score")

        print(metrics["F1"])

        print("\n")

        print("Confusion Matrix")

        print(metrics["Confusion Matrix"])

        print("\n")

        print("Classification Report")

        print(metrics["Classification Report"])

        print("=" * 70)

        predictions_np = predictions.numpy()

        probabilities_np = probabilities.numpy()
            ############################################################
    # Save Metrics
    ############################################################

    def save_metrics(
            self,
            evaluation,
            save_dir,
            filename="domain_metrics.csv"):

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        filepath = os.path.join(
            save_dir,
            filename
        )

        metrics = evaluation["metrics"]

        with open(
                filepath,
                "w",
                newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(["Metric", "Value"])

            writer.writerow(["Loss", metrics["Loss"]])

            writer.writerow(
                ["Domain Accuracy",
                 metrics["Domain Accuracy"]]
            )

            writer.writerow(
                ["Balanced Accuracy",
                 metrics["Balanced Accuracy"]]
            )

            writer.writerow(
                ["MCC",
                 metrics["MCC"]]
            )

            writer.writerow(
                ["Cohen Kappa",
                 metrics["Kappa"]]
            )

            writer.writerow(
                ["ROC-AUC",
                 metrics["ROC-AUC"]]
            )

            writer.writerow(
                ["Average Precision",
                 metrics["Average Precision"]]
            )

            writer.writerow([])

            writer.writerow(
                ["Per Domain Accuracy"]
            )

            for domain, value in metrics[
                    "Per Domain Accuracy"].items():

                writer.writerow(
                    [domain, value]
                )

        print(f"Metrics saved to {filepath}")

    ############################################################
    # Save Predictions
    ############################################################

    def save_predictions(
            self,
            evaluation,
            save_dir,
            filename="domain_predictions.csv"):

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        filepath = os.path.join(
            save_dir,
            filename
        )

        labels = evaluation["domain_labels"]

        predictions = evaluation["predictions"]

        with open(
                filepath,
                "w",
                newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(
                [
                    "Ground Truth Domain",
                    "Predicted Domain"
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

        print(f"Predictions saved to {filepath}")

    ############################################################
    # Save Features
    ############################################################

    def save_features(
            self,
            evaluation,
            save_dir,
            filename="domain_features.npy"):

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        filepath = os.path.join(
            save_dir,
            filename
        )

        np.save(
            filepath,
            evaluation["features"]
        )

        print(f"Features saved to {filepath}")

    ############################################################
    # Save Probabilities
    ############################################################

    def save_probabilities(
            self,
            evaluation,
            save_dir,
            filename="domain_probabilities.npy"):

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        filepath = os.path.join(
            save_dir,
            filename
        )

        np.save(
            filepath,
            evaluation["probabilities"]
        )

        print(f"Probabilities saved to {filepath}")

    ############################################################
    # Save Confusion Matrix
    ############################################################

    def save_confusion_matrix(
            self,
            evaluation,
            save_dir,
            filename="domain_confusion_matrix.csv"):

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        filepath = os.path.join(
            save_dir,
            filename
        )

        confusion = evaluation[
            "metrics"
        ]["Confusion Matrix"]

        np.savetxt(
            filepath,
            confusion,
            delimiter=",",
            fmt="%d"
        )

        print(
            f"Confusion matrix saved to {filepath}"
        )

    ############################################################
    # Save Complete Evaluation
    ############################################################

    def save_all(
            self,
            evaluation,
            save_dir):

        self.save_metrics(
            evaluation,
            save_dir
        )

        self.save_predictions(
            evaluation,
            save_dir
        )

        self.save_features(
            evaluation,
            save_dir
        )

        self.save_probabilities(
            evaluation,
            save_dir
        )

        self.save_confusion_matrix(
            evaluation,
            save_dir
        )

        print("\nAll evaluation results saved successfully.")
