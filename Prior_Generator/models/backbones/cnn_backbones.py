import torch.nn as nn
from torchvision import models as models_2d



class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained=True):
    model = models.resnet18(pretrained=True) 
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(weights=ResNet34_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=True):
    model = models_2d.resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024
