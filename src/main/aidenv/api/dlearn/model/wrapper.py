from aidenv.api.dlearn.model.logging import LoggableModel

class WrapperModel(LoggableModel):
    '''
    Abstract loggable deep model wrapping a module.
    '''

    def __init__(self, module):
        '''
        Create new wrapper model.
        module: torch module
        '''
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)