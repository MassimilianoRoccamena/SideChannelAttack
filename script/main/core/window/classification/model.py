from main.base.module.arch.resnet import ResNet1D
from main.core.model import SingleClassifierModel
from main.core.model import MultiClassifierModel

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

# C : unknown