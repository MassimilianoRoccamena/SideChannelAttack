import torch
import torch.nn as nn

class ModelA(nn.Module):
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)

    def forward(self, x):
        return self.model(x)