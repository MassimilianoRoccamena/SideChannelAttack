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
        self.log_dir = None

    def mount(self, *args, **kwargs):
        dataset = args[0]
        data_shape = dataset.data_shape()
        self.module.set_input_shape(data_shape)
        log_dir = args[1]
        self.log_dir = log_dir
        super().mount(*args[2:], **kwargs)

    def forward(self, x):
        return self.module(x)