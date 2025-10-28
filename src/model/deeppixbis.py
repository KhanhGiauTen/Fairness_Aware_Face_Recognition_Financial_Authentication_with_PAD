from typing import Any
import torch.nn as nn
import torchvision.models as models


class DeepPixBiS(nn.Module):
    def __init__(self):
        super(DeepPixBiS, self).__init__()
        densenet = models.densenet121(weights=None)
        self.features = densenet.features
        self.binary_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Any):
        feat = self.features(x)
        binary_out = self.binary_branch(feat)
        return binary_out
