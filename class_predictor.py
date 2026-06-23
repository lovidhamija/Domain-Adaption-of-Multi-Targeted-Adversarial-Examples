import torch.nn as nn


class ClassPredictor(nn.Module):

    def __init__(
        self,
        feature_dim=512,
        num_classes=100
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        logits = self.classifier(x)

        return logits