import torch
from torch.nn import GRU

from aidenv.api.dlearn.module.encoder import WrapperEncoder

class GRUEncoder(WrapperEncoder):
    '''
    GRU encoder.
    '''

    def encoder_module(self, input_shape):
        return GRU(*input_shape, **self.kwargs)

class GRU1(GRUEncoder):
    '''
    GRU with encoding function of only the last output.
    '''

    def __init__(self, **kwargs):
        kwargs['final_size'] = kwargs['hidden_size']
        super().__init__(**kwargs)

    def forward(self, x):
        outputs, last_states = self.module(x)
        y = outputs[:,-1,:]
        y = super().forward(y)
        return y