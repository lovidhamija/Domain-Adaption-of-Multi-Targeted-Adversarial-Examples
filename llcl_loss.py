import torch
import torch.nn as nn
import torch.nn.functional as F


class LLCLLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def pearson_corr(

        self,

        source,

        target
    ):

        source = source - source.mean(
            dim=1,
            keepdim=True
        )

        target = target - target.mean(
            dim=1,
            keepdim=True
        )

        numerator = torch.sum(

            source * target,

            dim=1
        )

        denominator = (

            torch.sqrt(

                torch.sum(
                    source ** 2,
                    dim=1
                )
            )

            *

            torch.sqrt(

                torch.sum(
                    target ** 2,
                    dim=1
                )
            )

            + 1e-8
        )

        corr = numerator / denominator

        return corr

    def forward(

        self,

        source_logits,

        target_logits,

        class_weights
    ):

        source_prob = F.softmax(

            source_logits,

            dim=1
        )

        target_prob = F.softmax(

            target_logits,

            dim=1
        )

        corr = self.pearson_corr(

            source_prob,

            target_prob
        )

        pseudo_labels = torch.argmax(

            source_prob,

            dim=1
        )

        weights = class_weights[
            pseudo_labels
        ]

        source_sum = torch.sum(

            source_prob,

            dim=1
        )

        target_sum = torch.sum(

            target_prob,

            dim=1
        )

        llcl = -weights * corr * (

            source_sum
            *
            target_sum
        )

        loss = llcl.mean()

        return loss