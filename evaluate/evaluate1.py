
import os
import csv
import torch
import numpy as np

from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

from evaluation.metrics import ClassificationMetrics


class Evaluator:

    def __init__(
        self,
        encoder,
        classifier,
        device,
        num_classes
    ):

        self.encoder = encoder
        self.classifier = classifier

        self.device = device

        self.criterion = CrossEntropyLoss()

        self.metrics = ClassificationMetrics(
            num_classes=num_classes
        )

    ##########################################################
    # Classification Evaluation
    ##########################################################

    @torch.no_grad()
    def evaluate(
        self,
        dataloader
    ):

        self.encoder.eval()
        self.classifier.eval()

        running_loss = 0.0

        total = 0

        correct = 0

        all_labels = []

        all_predictions = []

        all_probabilities = []

        all_features = []

        for images, labels in tqdm(dataloader):

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            ##################################################
            # Forward
            ##################################################

            features = self.encoder(
                images
            )

            logits = self.classifier(
                features
            )

            loss = self.criterion(
                logits,
                labels
            )

            ##################################################
            # Prediction
            ##################################################

            probabilities = softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            ##################################################
            # Statistics
            ##################################################

            running_loss += loss.item() * labels.size(0)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            ##################################################
            # Store
            ##################################################

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_features.extend(
                features.cpu().numpy()
            )

        ######################################################
        # Final Results
        ######################################################

        avg_loss = running_loss / total

        results = self.metrics.evaluate(

            y_true=np.array(
                all_labels
            ),

            y_pred=np.array(
                all_predictions
            ),

            probabilities=np.array(
                all_probabilities
            )
        )

        results["Loss"] = avg_loss

        results["Top1 Accuracy"] = correct / total

        return {

            "metrics": results,

            "features": np.array(
                all_features
            ),

            "labels": np.array(
                all_labels
            ),

            "predictions": np.array(
                all_predictions
            ),

            "probabilities": np.array(
                all_probabilities
            )

        }

    ##########################################################
    # Source Evaluation
    ##########################################################

    def evaluate_source(
        self,
        source_loader
    ):

        print("\nEvaluating Source Domain...")

        return self.evaluate(
            source_loader
        )

    ##########################################################
    # Target Evaluation
    ##########################################################

    def evaluate_target(
        self,
        target_loader
    ):

        print("\nEvaluating Target Domain...")

        return self.evaluate(
            target_loader
        )
