from .clip import clip
from .classifier_heads import build_classifier_head
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from numbers import Number
import math


def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}


class CLIPModel(nn.Module):
    def __init__(
        self,
        name,
        num_classes=1,
        classifier_head="linear",
        uncertainty_weighting=False,
        uncertainty_weight_scale=1.0,
        uncertainty_weight_mode="residual",
        vib_inference_mode="sample",
    ):
        super(CLIPModel, self).__init__()
        self.k = 256
        self.uncertainty_weighting = uncertainty_weighting
        self.uncertainty_weight_scale = uncertainty_weight_scale
        self.uncertainty_weight_mode = uncertainty_weight_mode
        self.vib_inference_mode = vib_inference_mode
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.fc_1 = nn.Linear(768, 1024)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 2*self.k)
        self.decode = build_classifier_head(classifier_head, self.k, out_dim=num_classes)
        self.dropout = nn.Dropout(0.5)

    def extract_features(self, x):
        features = self.model.encode_image(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features

    def forward_from_features(self, features):
        num_sample = 1
        features = self.dropout(features)
        statistics = self.fc_1(features)
        statistics = self.relu(statistics)
        statistics = self.fc_2(statistics)
        statistics = self.relu(statistics)
        statistics = self.fc_3(statistics)
        mu = statistics[:,:self.k]
        std = F.softplus(statistics[:,self.k:]-5,beta=1)

        if self.vib_inference_mode == "sample":
            encoding = self.reparametrize_n(mu,std,num_sample)
            if self.uncertainty_weighting:
                weight = torch.exp(-self.uncertainty_weight_scale * std)
                if self.uncertainty_weight_mode == "direct":
                    encoding = weight * encoding
                elif self.uncertainty_weight_mode == "residual":
                    encoding = mu + weight * (encoding - mu)
                else:
                    raise ValueError(f"Unsupported uncertainty_weight_mode: {self.uncertainty_weight_mode}")
            logit = self.decode(encoding)
        elif self.vib_inference_mode == "mean":
            logit = self.decode(mu)
        elif self.vib_inference_mode == "marginal":
            logit = self.marginalized_logit(mu, std)
        else:
            raise ValueError(f"Unsupported vib_inference_mode: {self.vib_inference_mode}")

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def forward(self, x):
        features = self.extract_features(x)
        return self.forward_from_features(features)

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def marginalized_logit(self, mu, std):
        if not hasattr(self.decode, "weight"):
            raise ValueError("marginal VIB inference requires a linear classifier head with weight and bias.")
        logit_mean = F.linear(mu, self.decode.weight, self.decode.bias)
        logit_var = F.linear(std.pow(2), self.decode.weight.pow(2), None).clamp_min(0.0)
        scale = torch.sqrt(1.0 + (math.pi / 8.0) * logit_var)
        return logit_mean / scale


class NoVIBCLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, classifier_head="dropout_linear"):
        super(NoVIBCLIPModel, self).__init__()
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.feature_dim = CHANNELS[name]
        self.decode = build_classifier_head(classifier_head, self.feature_dim, out_dim=num_classes)

    def extract_features(self, x):
        features = self.model.encode_image(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logit = self.decode(features)
        dummy_std = torch.zeros(features.size(0), 1, device=features.device, dtype=features.dtype)
        return (features, dummy_std), logit
