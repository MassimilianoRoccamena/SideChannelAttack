from aidenv.api.dlearn.model import SingleClassifierModel
from aidenv.api.dlearn.model import MultiClassifierModel
from aidenv.api.dlearn.module.arch.resnet import ResNetEncoder \
    as ResNet
from aidenv.api.dlearn.module.arch.lstm import LSTMEncoder \
    as LSTM

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
