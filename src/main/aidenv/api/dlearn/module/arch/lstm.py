from torch.nn import LSTM

from aidenv.api.dlearn.module.encoder import EncoderModule

class LSTMEncoder(EncoderModule):
    '''
    LSTM encoder.
    '''

    def __init__(self, **kwargs):
        encoding_size = kwargs.pop('encoding_size')
        use_final_do = kwargs.pop('use_final_do')
        final_do_val = kwargs.pop('final_do_val')
        super().__init__(encoding_size, use_final_do,
                            final_size=kwargs['hidden_size'], final_do_val=final_do_val)

        self.kwargs = kwargs

    def set_input_shape(self, input_shape):
        if input_shape is None:
            self.module = None
        else:
            self.module = LSTM(*input_shape, **self.kwargs)
        super().set_input_shape(input_shape)

    def forward(self, x):
        y = self.module(x)[1][0][-1]
        y = super().forward(y)
        return y