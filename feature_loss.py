import torch
import torch.nn as nn


class FeatureLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

    def forward(

        self,

        teacher_features,

        student_features
    ):

        loss = self.mse(

            student_features,

            teacher_features
        )

        return loss