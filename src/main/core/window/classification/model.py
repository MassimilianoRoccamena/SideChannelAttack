from main.bridge.model import SingleClassifierModel
from main.bridge.model import MultiClassifierModel
from main.bridge.module.arch.resnet import ResNet

# A : convolutional

class SingleA(SingleClassifierModel):
    '''
    Convolution based single classifier
    '''

    pass

class MultiA(MultiClassifierModel):
    '''
    Convolution based model with (voltage, frequency)
    labelling
    '''

    pass

class VoltageA(SingleA):
    '''
    Convolution model with voltage labelling
    '''

    pass

class FrequencyA(SingleA):
    '''
    Convolution model with frequency labelling
    '''

    pass

# B : recurrent

# C : attention