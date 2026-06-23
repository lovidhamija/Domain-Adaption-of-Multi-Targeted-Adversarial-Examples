import torch
import torch.nn as nn


class TeacherNetwork(nn.Module):

    def __init__(self, backbone):

        super().__init__()

        self.encoder = nn.Sequential(

            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,

            backbone.avgpool
        )

        self.classifier = backbone.fc

    def forward(self, x):

        features = self.encoder(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        logits = self.classifier(
            features
        )

        return features, logits