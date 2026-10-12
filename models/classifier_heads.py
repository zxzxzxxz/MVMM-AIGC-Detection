import torch.nn as nn
import torch.nn.functional as F


class DropoutLinearHead(nn.Module):
    def __init__(self, in_dim, out_dim=1, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        linear = nn.Linear(in_dim, out_dim)
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone())

    def forward(self, x):
        x = self.dropout(x)
        return F.linear(x, self.weight, self.bias)


def build_classifier_head(mode, in_dim, out_dim=1):
    if mode == "linear":
        return DropoutLinearHead(in_dim, out_dim=out_dim, dropout=0.0)
    if mode == "dropout_linear":
        return DropoutLinearHead(in_dim, out_dim=out_dim, dropout=0.2)
    raise ValueError(f"Unsupported classifier head mode: {mode}")
