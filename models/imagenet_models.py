from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from .classifier_heads import build_classifier_head

from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn


model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vit_b_16': vit_b_16,
    'vit_b_32': vit_b_32,
    'vit_l_16': vit_l_16,
    'vit_l_32': vit_l_32
}


CHANNELS = {
    "resnet50" : 2048,
    "vit_b_16" : 768,
}



class ImagenetModel(nn.Module):
    def __init__(self, name, num_classes=1, classifier_head="linear"):
        super(ImagenetModel, self).__init__()
        self.model = model_dict[name](pretrained=True)
        self.fc = build_classifier_head(classifier_head, CHANNELS[name], out_dim=num_classes)


    def forward(self, x):
        feature = self.model(x)["penultimate"]
        logit = self.fc(feature)
        dummy_std = torch.zeros_like(feature)
        return (feature, dummy_std), logit
