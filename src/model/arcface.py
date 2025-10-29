from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet101', pretrained=False):
        super().__init__()
        
        assert backbone_name == 'resnet101', "This version supports only resnet101"

        if pretrained:
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet101(weights=None)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = resnet.fc.in_features 

    def forward(self, x: Any):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x


class ArcFaceModel(nn.Module):
    def __init__(self, num_classes=1000, feature_dim=512, pretrained=False):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(
            backbone_name='resnet101', 
            pretrained=pretrained
        )

        self.fc = nn.Linear(self.feature_extractor.output_dim, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        
    def forward(self, x: Any, labels=None):
        x = self.feature_extractor(x)
        x = self.fc(x)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)