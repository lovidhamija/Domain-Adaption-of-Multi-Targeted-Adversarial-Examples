0 = Clean
1 = FGSM
2 = PGD
3 = CW
4 = Patch
import torch.nn as nn


class DomainDiscriminator(nn.Module):

    def __init__(
        self,
        feature_dim=512,
        num_domains=5
    ):

        super().__init__()

        self.discriminator = nn.Sequential(

            nn.Linear(
                feature_dim,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_domains
            )
        )

    def forward(self, x):

        return self.discriminator(x)