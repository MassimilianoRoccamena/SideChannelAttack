import torch.nn as nn

from aidenv.api.dlearn.module.config import CoreModule

class EncoderModule(CoreModule):
    '''
    Neural encoder module.
    '''

    def __init__(self, encoding_size, use_final_do, final_size=None, final_do_val=0.5):
        '''
        Create new neural encoder.
        encoding_size: size of the encoder layer
        use_final_do: wheter to use dropout before encoder layer
        final_do_val: dropout value between final and encoder layer
        final_size: size of the final layer
        '''
        super().__init__()
        self.encoding_size = encoding_size
        self.use_final_do = use_final_do
        self.final_do_val = final_do_val
        self.final_do = nn.Dropout(p=final_do_val)
        self.set_final_size(final_size)
        
    def set_final_size(self, final_size):
        '''
        Set size of the final layer.
        final_size: size of the final layer
        '''
        if final_size is None:
            self.final_size = None
            self.encoding = None
        else:
            self.final_size = final_size
            self.encoding = nn.Linear(final_size, self.encoding_size)

    def forward(self, x):
        if self.use_final_do:
            x = self.final_do(x)
        return self.encoding(x)