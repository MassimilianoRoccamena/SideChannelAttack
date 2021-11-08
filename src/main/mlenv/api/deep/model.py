from pytorch_lightning import LightningModule

from main.mlenv.api.config import CoreObject
from main.mlenv.api.deep.config import build_model_object1
from main.mlenv.api.deep.module.classifier import SingleClassifier
from main.mlenv.api.deep.module.classifier import MultiClassifier

class CoreModel(LightningModule, CoreObject):
    ''''
    Abstract core model
    '''

    def set_learning(self, loss, optimizer, scheduler=None):
        '''
        Set model learning parameters.
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self, batch, batch_index):
        '''
        Basic loss computation on a batch.
        '''
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_index):
        return self.step(batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.step(batch, batch_index)

    def test_step(self, batch, batch_index):
        return self.step(batch, batch_index)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]

    def mount_from_dataset(self, dataset):
        '''
        Mount the part of the model which is function of some data.
        dataset: dataset object
        '''
        raise NotImplementedError

class WrapperModel(CoreModel):
    '''
    Abstract core model wrapping a module
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

# classification

class ClassifierModel(WrapperModel):
    '''
    Abstract model wrapping a classifier
    '''

    @classmethod
    def build_kwargs(cls, config, core_prompt):
        encoder = build_model_object1(config.encoder, core_prompt)
        config = cls.update_kwargs(config, encoder=encoder)
        return config

    def mount_from_dataset(self, dataset):
        self.module.set_labels(dataset.all_labels())

class SingleClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a single classifier
    '''

    def __init__(self, encoder):
        super().__init__(SingleClassifier(encoder))

class MultiClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a multiple classifier
    '''

    def __init__(self, encoder):
        super().__init__(MultiClassifier(encoder))
        raise NotImplementedError