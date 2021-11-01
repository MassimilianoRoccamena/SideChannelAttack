import torch

from main.core.model import ConfigModel

class A(ConfigModel):
    '''
    Classic vision-based model
    '''

    def __init__(self):
        #self.model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)
        pass

    @classmethod
    def parse_args(cls, config, core_nodes):
        return []

    def forward(self, x):
        return self.model(x)

class MixedA(A):
    pass

class VoltageA(A):
    pass

class FrequencyA(A):
    pass