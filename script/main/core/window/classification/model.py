from main.core.model import ConfigModel

class A(ConfigModel):
    '''
    Convolution based model
    '''

    def __init__(self):
        #self.model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)
        pass

    @classmethod
    def config_args(cls, config, core_nodes):
        return []

    def forward(self, x):
        return x

class MixedA(A):
    '''
    Convolution model with voltage, frequency labelling
    '''

    pass

class VoltageA(A):
    '''
    Convolution model with voltage labelling
    '''

    pass

class FrequencyA(A):
    '''
    Convolution model with frequency labelling
    '''

    pass