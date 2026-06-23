import copy
import torch.nn as nn


class SharedEncoder(nn.Module):

    def __init__(
        self,
        teacher_encoder
    ):

        super().__init__()

        self.encoder = copy.deepcopy(
            teacher_encoder
        )

    def forward(self, x):

        features = self.encoder(x)

        features = features.flatten(
            start_dim=1
        )

        return features
