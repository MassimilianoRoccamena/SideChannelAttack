import torch

def get_A():
    return torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)