import torch.nn as nn
import torch


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier Initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )  # Kaiming Initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)
