import torch
from torch.nn import TransformerEncoderLayer

from aidenv.api.dlearn.module.encoder import WrapperEncoder

class Transformer(WrapperEncoder):
    '''
    Transformer encoder.
    '''

    def __init__(self, **kwargs):
        kwargs['final_size'] = kwargs['dim_feedforward']
        super().__init__(**kwargs)

    def encoder_module(self, input_shape):
        return TransformerEncoderLayer(*input_shape, **self.kwargs)

    def forward(self, x):
        y = self.module(x)
        y = y.view(y.shape[:-1])
        return super().forward(y)