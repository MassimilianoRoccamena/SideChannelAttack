import torch
from torch.nn import Conv1d, LSTM

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
        final_size = kwargs['final_size']
        kernel_size = kwargs.pop('kernel_size')
        stride = kwargs.pop('stride')
        kwargs['final_size'] = final_size
        super().__init__(**kwargs)
        self.conv = Conv1d(kwargs['hidden_size'], final_size, kernel_size, stride=stride)
    
    def forward(self, x):
        #y = self.module(x)[1][0][-1]
        outputs, last_states = self.module(x)
        outputs = outputs.permute(0,2,1)
        y = self.conv(outputs)
        #y = torch.flatten(outputs, start_dim=1)
        y = y.mean(-1)
        y = super().forward(y)
        return y