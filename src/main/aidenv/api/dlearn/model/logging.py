from aidenv.api.dlearn.model.basic import DeepModel
from aidenv.api.dlearn.learning import LoggableLoss

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
        self.loggables['train'].update({'loss' : LoggableLoss(progr_bar=False)})
        self.loggables['valid'].update({'loss' : LoggableLoss(progr_bar=False)})
        self.loggables['test'].update({'loss' : LoggableLoss(progr_bar=False)})

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
                loggable.mount(self, *args, **kwargs)

    # steps

    def compute_epoch_step(self, batch, prefix, include_loss=False):
        pred, target, loss = super().step_batch(batch)

        outputs = {}
        for log_name, loggable in self.loggables[prefix].items():
            log_name = f'{prefix}/{log_name}'
            loggable.log(outputs, log_name, pred, target, loss)

        if include_loss:
            outputs['loss'] = loss
        
        return outputs

    def training_step(self, batch, batch_index):
        return self.compute_epoch_step(batch, 'train', include_loss=True)

    def validation_step(self, batch, batch_index):
        return self.compute_epoch_step(batch, 'valid')

    def test_step(self, batch, batch_index):
        return self.compute_epoch_step(batch, 'test')

    # epoch end

    def compute_epoch_end(self, outputs, prefix):
        for log_name, loggable in self.loggables[prefix].items():
            if prefix != 'test':
                loggable.on_epoch_end(outputs, prefix=prefix, log_name=log_name,
                                        epoch=self.current_epoch)
            else:
                loggable.on_epoch_end(outputs, prefix=prefix, log_name=log_name)

    def training_epoch_end(self, outputs):
        self.compute_epoch_end(outputs, 'train')
        
    def validation_epoch_end(self, outputs):
        self.compute_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs):
        self.compute_epoch_end(outputs, 'test')
