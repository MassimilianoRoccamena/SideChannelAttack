from torch.nn import Module as TorchModule

from aidenv.api.config import CoreObject

class CoreModule(TorchModule, CoreObject):
    '''
    Abstract core neural module.
    '''

    def set_input_shape(self, input_shape):
        '''
        Set module input shape.
        input_size: input shape of the module
        '''
        self.input_shape = input_shape