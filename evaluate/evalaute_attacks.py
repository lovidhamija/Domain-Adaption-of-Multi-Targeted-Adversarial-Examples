"""
===========================================================
evaluate_attacks.py

Research Evaluation Module

Supports

1. Clean Evaluation
2. FGSM
3. PGD
4. CW
5. Patch Attack
6. AutoAttack

===========================================================
"""

import os
import csv
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

import torchattacks

from evaluation.metrics import ClassificationMetrics


class AttackEvaluator:

    def __init__(
            self,
            model,
            device,
            num_classes=100,
            eps=8/255,
            alpha=2/255,
            pgd_steps=10,
            cw_steps=50):

        self.model = model.to(device)

        self.device = device

        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.metrics = ClassificationMetrics(
            num_classes=num_classes
        )

        ####################################################
        # Attack Parameters
        ####################################################

        self.eps = eps

        self.alpha = alpha

        self.pgd_steps = pgd_steps

        self.cw_steps = cw_steps

        ####################################################
        # Build Attacks
        ####################################################

        self.build_attacks()
      ####################################################
    # Build Attacks
    ####################################################

    def build_attacks(self):

        self.fgsm = torchattacks.FGSM(

            self.model,

            eps=self.eps

        )

        self.pgd = torchattacks.PGD(

            self.model,

            eps=self.eps,

            alpha=self.alpha,

            steps=self.pgd_steps,

            random_start=True

        )

        self.cw = torchattacks.CW(

            self.model,

            c=1,

            kappa=0,

            steps=self.cw_steps

        )

        ################################################
        # Optional AutoAttack
        ################################################

        try:

            self.autoattack = torchattacks.AutoAttack(

                self.model,

                norm="Linf",

                eps=self.eps,

                version="standard"

            )

            print("AutoAttack Loaded.")

        except Exception:

            self.autoattack = None

            print("AutoAttack Not Available.") 
  ####################################################
    # Patch Attack
    ####################################################

    def patch_attack(

            self,

            images,

            patch_size=8,

            value=1.0):

        patched = images.clone()

        patched[
            :,
            :,
            0:patch_size,
            0:patch_size
        ] = value

        return patched
          ####################################################
    # Generate Attack
    ####################################################

    def generate_attack(

            self,

            images,

            labels,

            attack):

        attack = attack.lower()

        if attack == "clean":

            return images

        elif attack == "fgsm":

            return self.fgsm(

                images,

                labels

            )

        elif attack == "pgd":

            return self.pgd(

                images,

                labels

            )

        elif attack == "cw":

            return self.cw(

                images,

                labels

            )

        elif attack == "patch":

            return self.patch_attack(

                images

            )

        elif attack == "autoattack":

            if self.autoattack is None:

                raise RuntimeError(

                    "AutoAttack unavailable."

                )

            return self.autoattack(

                images,

                labels

            )

        else:

            raise ValueError(

                f"Unknown attack {attack}"

            )
            ####################################################
    # Evaluate One Attack
    ####################################################

    def evaluate_attack(

            self,

            dataloader,

            attack="clean"):

        self.model.eval()

        running_loss = 0.0

        total = 0

        correct = 0

        feature_list = []

        probability_list = []

        prediction_list = []

        label_list = []

        logits_list = []
                ####################################################
        # Evaluation Loop
        ####################################################

        for images, labels in tqdm(
                dataloader,
                desc=f"{attack.upper()} Evaluation"):

            images = images.to(self.device)

            labels = labels.to(self.device)

            ################################################
            # Generate Adversarial Images
            ################################################

            if attack.lower() == "clean":

                adv_images = images

            else:

                adv_images = self.generate_attack(

                    images,

                    labels,

                    attack

                )

            ################################################
            # Forward Pass
            ################################################

            outputs = self.model(adv_images)

            ################################################
            # Wrapper Compatibility
            ################################################

            if isinstance(outputs, tuple):

                if len(outputs) == 2:

                    features, logits = outputs

                else:

                    logits = outputs[-1]

                    features = outputs[0]

            else:

                logits = outputs

                features = None

            ################################################
            # Compute Loss
            ################################################

            loss = self.criterion(

                logits,

                labels

            )

            running_loss += (

                loss.item()

                * labels.size(0)

            )

            ################################################
            # Softmax Probabilities
            ################################################

            probabilities = torch.softmax(

                logits,

                dim=1

            )

            ################################################
            # Predictions
            ################################################

            predictions = torch.argmax(

                probabilities,

                dim=1

            )

            ################################################
            # Accuracy
            ################################################

            correct += (

                predictions == labels

            ).sum().item()

            total += labels.size(0)

            ################################################
            # Save Features
            ################################################

            if features is not None:

                feature_list.append(

                    features.detach().cpu()

                )

            ################################################
            # Save Outputs
            ################################################

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
                    ####################################################
        # Concatenate Results
        ####################################################

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

        if len(feature_list) > 0:

            features = torch.cat(

                feature_list,

                dim=0

            )

            features_np = features.numpy()

        else:

            features_np = None

        ####################################################
        # Average Loss
        ####################################################

        average_loss = running_loss / total

        ####################################################
        # Robust Accuracy
        ####################################################

        robust_accuracy = correct / total

        ####################################################
        # Top-5 Accuracy
        ####################################################

        top5_accuracy = self.metrics.top_k_accuracy(

            logits,

            labels,

            k=5

        )

        ####################################################
        # Convert to NumPy
        ####################################################

        labels_np = labels.numpy()

        predictions_np = predictions.numpy()

        probabilities_np = probabilities.numpy()

        ####################################################
        # Classification Metrics
        ####################################################

        results = self.metrics.evaluate(

            y_true=labels_np,

            y_pred=predictions_np,

            probabilities=probabilities_np

        )

        ####################################################
        # Extra Statistics
        ####################################################

        results["Attack"] = attack

        results["Loss"] = average_loss

        results["Robust Accuracy"] = robust_accuracy

        results["Top5 Accuracy"] = top5_accuracy

        ####################################################
        # Attack Success Rate (ASR)
        ####################################################

        if attack.lower() == "clean":

            attack_success_rate = 0.0

        else:

            attack_success_rate = 1.0 - robust_accuracy

        results["Attack Success Rate"] = attack_success_rate

        ####################################################
        # Return Evaluation
        ####################################################

        evaluation = {

            "metrics": results,

            "features": features_np,

            "logits": logits.numpy(),

            "probabilities": probabilities_np,

            "predictions": predictions_np,

            "labels": labels_np

        }

        return evaluation
 
