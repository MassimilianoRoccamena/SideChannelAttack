import copy
import torch
from torchmetrics import Metric, MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix

from aidenv.api.dlearn.model.basic import DeepModel
from aidenv.api.dlearn.learning.logging import LoggableAdvanced

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
        self.loggables_adv = { 'train' : {}, 'valid' : {}, 'test' : {} }

    def add_loggables(self, loggables, sets):
        '''
        Add loggable objects to the model.
        loggables: named dict of loggable objects
        prefix: prefix of the log name
        '''
        for name,loggable in loggables.items():
            for s in sets[name]:
                if isinstance(loggable, LoggableAdvanced):
                    self.loggables_adv[s].update({name : loggable})
                else:
                    self.loggables[s].update({name : loggable})

    def mount(self, *args, **kwargs):
        #accuracy = Accuracy(num_classes=9, average='macro')
        #f1 = F1(num_classes=9, average='macro')
        #self.metrics = MetricCollection({'train/accuracy':accuracy, 'train/f1':f1})
        #self.cmat = ConfusionMatrix(num_classes=9, normalize='true')
        for prefix, sublogs in self.loggables.items():
            metrics = {}
            for log_name, loggable in sublogs.items():
                loggable.mount(self, *args, **kwargs)
                metrics[log_name] = loggable.metric

            metrics = MetricCollection(metrics)
            if prefix == 'train':
                self.metrics_train = metrics.clone(prefix='train/')
            elif prefix == 'valid':
                self.metrics_valid = metrics.clone(prefix='valid/')
            elif prefix == 'test':
                self.metrics_test = metrics.clone(prefix='test/')
            else:
                raise RuntimeError(f'invalid prefix {prefix}')

        for prefix, sublogs in self.loggables_adv.items():
            for log_name, loggable in sublogs.items():
                loggable.mount(self, *args, **kwargs)

    # step end methods

    def loggables_adv_step_end(self, outputs, prefix):
        for log_name, loggable in self.loggables_adv[prefix].items():
            log_name = f'{prefix}/{log_name}'
            loggable.log(outputs, log_name)

    def training_step_end(self, outputs):
        #self.log('train/loss', outputs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        #self.metrics(outputs['prediction'], outputs['target'])
        #self.log_dict(self.metrics, on_step=True, on_epoch=True, sync_dist=True)
        #self.cmat(outputs['prediction'], outputs['target'])

        #self.metrics_train(outputs['prediction'], outputs['target'])
        #self.log_dict(self.metrics_train, on_step=True, on_epoch=True, sync_dist=True)

        #self.compute_step_end(outputs, 'train')

        self.log(f'train/loss', outputs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.metrics_train(outputs['prediction'], outputs['target'])
        self.log_dict(self.metrics_train, on_step=True, on_epoch=True, sync_dist=True)
        self.loggables_adv_step_end(outputs, 'train')

    def validation_step_end(self, outputs):
        #self.log('valid/loss', outputs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        #self.metrics_valid(outputs['prediction'], outputs['target'])
        #self.log_dict(self.metrics_valid, on_step=True, on_epoch=True, sync_dist=True)

        #self.compute_step_end(outputs, 'valid')

        self.log(f'valid/loss', outputs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.metrics_valid(outputs['prediction'], outputs['target'])
        self.log_dict(self.metrics_valid, on_step=True, on_epoch=True, sync_dist=True)
        self.loggables_adv_step_end(outputs, 'valid')

    def test_step_end(self, outputs):
        #self.log('test/loss', outputs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        #self.metrics_test(outputs['prediction'], outputs['target'])

        #self.compute_step_end(outputs, 'test')

        self.log(f'test/loss', outputs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.metrics_test(outputs['prediction'], outputs['target'])
        self.log_dict(self.metrics_test, on_step=True, on_epoch=True, sync_dist=True)
        self.loggables_adv_step_end(outputs, 'test')
    
    # epoch end methods

    def compute_epoch_end(self, outputs, prefix):
        for log_name, loggable in self.loggables_adv[prefix].items():
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
