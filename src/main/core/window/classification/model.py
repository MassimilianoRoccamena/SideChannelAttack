from main.bridge.model import SingleClassifierModel
from main.bridge.model import MultiClassifierModel
from main.bridge.module.arch.resnet import ResNet

# Basic : encoder + classifier

class SingleBasic(SingleClassifierModel):
    '''
    Abstract encoder + single classifier model
    '''

    pass

class MultiBasic(MultiClassifierModel):
    '''
    Encoder + multiple classifier model
    '''

    pass

class VoltageBasic(SingleBasic):
    '''
    Encoder + voltage classifier model
    '''

    pass

class FrequencyBasic(SingleBasic):
    '''
    Encoder + frequency classifier model
    '''

    pass
