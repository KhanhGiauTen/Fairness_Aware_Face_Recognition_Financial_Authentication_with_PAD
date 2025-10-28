from typing import Any
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet34', output_dim=512, pretrained=False):
        super().__init__()
        assert backbone_name == 'resnet34', "Only resnet34 is supported in this helper"
        resnet = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        actual_in_features = 512
        if output_dim != actual_in_features:
            self.fc = nn.Linear(actual_in_features, output_dim)
        else:
            self.fc = nn.Identity()

    def forward(self, x: Any):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class ArcFaceModel(nn.Module):
    def __init__(self, num_classes=1000, feature_dim=512):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(output_dim=feature_dim, pretrained=False)

    def forward(self, x: Any, labels=None):
        features = self.feature_extractor(x)
        return F.normalize(features)
