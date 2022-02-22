import torch
from torch.nn import LSTM

from aidenv.api.dlearn.module.encoder import WrapperEncoder

class LSTMEncoder(WrapperEncoder):
    '''
    LSTM encoder.
    '''

    def encoder_module(self, input_shape):
        return LSTM(*input_shape, **self.kwargs)

class LSTM1(LSTMEncoder):
    '''
    LSTM with encoding function of only the last output.
    '''

    def __init__(self, **kwargs):
        kwargs['final_size'] = kwargs['hidden_size']
        super().__init__(**kwargs)

    def forward(self, x):
        outputs, last_states = self.module(x)
        y = outputs[:,-1,:]
        y = super().forward(y)
        return y

class LSTM2(LSTMEncoder):
    '''
    LSTM with encoding function of the entire output last output.
    '''

    def __init__(self, **kwargs):
        kwargs['final_size'] = kwargs['hidden_size'] * 500
        super().__init__(**kwargs)
    
    def forward(self, x):
        #y = self.module(x)[1][0][-1]
        outputs, last_states = self.module(x)
        y = torch.flatten(outputs, start_dim=1)
        y = super().forward(y)
        return y