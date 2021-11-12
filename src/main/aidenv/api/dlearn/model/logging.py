from aidenv.api.dlearn.model.basic import DeepModel

class LoggableObject:
    '''
    Abstract loggable object of a model.
    '''

    def set_model(self, model):
        '''
        Set parent model for the loggable object.
        model: parent model
        '''
        self.model = model

    def __call__(self, *args, **kwargs):
        '''
        Compute the value of the loggable.
        '''
        raise NotImplementedError

    def on_epoch_end(self, *args, **kwargs):
        '''
        Handle model training epoch end.
        '''
        raise NotImplementedError

class LoggableScalar(LoggableObject):
    '''
    AbstractLoggable scalar number.
    '''

    def on_epoch_end(self, *args, **kwargs):
        pass

class LoggableLoss(LoggableScalar):
    '''
    Loggable loss value.
    '''

    def __call__(self, *args, **kwargs):
        return kwargs['loss']

class LoggableFigure(LoggableObject):
    '''
    Abstract loggable figure.
    '''

    def on_epoch_end(self, *args, **kwargs):
        self.model.logger.log_figure(*args, **kwargs)

    def draw(self, *args, **kwargs):
        '''
        Draw the figure.
        '''
        raise NotImplementedError

# model

class LoggableModel(DeepModel):
    '''
    Abstract deep model which logs some objects.
    '''

    def __init__(self):
        '''
        Create new loggable deep model.
        '''
        super().__init__()
        self.loggables = { 'train' : {}, 'valid' : {}, 'test' : {} }
        #self.loggables['train'].update({'train/loss' : LoggableLoss()})
        #self.loggables['valid'].update({'valid/loss' : LoggableLoss()})
        #self.loggables['test'].update({'test/loss' : LoggableLoss()})

    # setup

    def add_loggables(self, loggables, prefix):
        '''
        Add loggable objects to the moddel.
        loggables: named dict of loggable objects
        prefix: prefix of the log name
        '''
        self.loggables[prefix].update(loggables)

    def mount(self, *args, **kwargs):
        for prefix, sublogs in self.loggables.items():
            for log_name, loggable in sublogs.items():
                loggable.set_model(self)

    # steps

    def compute_epoch_step(self, batch, prefix, include_loss=False):
        pred, target, loss = super().step_batch(batch)

        logs = {}
        for log_name, loggable in self.loggables[prefix].items():
            log_name = f'{prefix}/{log_name}'
            logs[log_name] = loggable(pred, target, loss=loss)

        output = {}
        output.update(logs)

        self.log_dict(logs,
                        on_step=True, on_epoch=True,
                        sync_dist=True, prog_bar=True)

        if include_loss:
            output['loss'] = loss

        return output

    def training_step(self, batch, batch_index):
        return self.compute_epoch_step(batch, 'train', include_loss=True)

    def validation_step(self, batch, batch_index):
        return self.compute_epoch_step(batch, 'valid')

    def test_step(self, batch, batch_index):
        return self.compute_epoch_step(batch, 'test')

    # epoch end

    def compute_epoch_end(self, outputs, prefix):
        for log_name, loggable in self.loggables[prefix].items():
            loggable.on_epoch_end(outputs, prefix=prefix, log_name=log_name)

    def training_epoch_end(self, outputs):
        self.compute_epoch_end(outputs, 'train')
        
    def validation_epoch_end(self, outputs):
        self.compute_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs):
        self.compute_epoch_end(outputs, 'test')
