from aidenv.api.dlearn.model import SingleClassifierModel, SingleClassifierAdvancedModel
from aidenv.api.dlearn.model import MultiClassifierModel
from aidenv.api.dlearn.module.arch.vgg import VGG
from aidenv.api.dlearn.module.arch.resnet import ResNet
from aidenv.api.dlearn.module.arch.lstm import LSTM1, LSTM2
from aidenv.api.dlearn.module.arch.gru import GRU1
#from aidenv.api.dlearn.module.arch.custom import RecurrentAttention

# Basic : encoder + classifier

class SingleBasic(SingleClassifierModel):
    '''
    Abstract encoder + single classifier model
    '''

    pass

class SingleAdvanced(SingleClassifierAdvancedModel):
    '''
    Abstract encoder + fully connected single classifier model
    '''

    pass

class MultiBasic(MultiClassifierModel):
    '''
    Encoder + multiple classifier model
    '''

    pass
