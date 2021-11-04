from pytorch_lightning import LightningModule

from main.base.app.params import MODEL_MODULE
from main.base.app.config import ConfigObject
from main.base.app.config import config_core_object1
from main.base.module.classifier import SingleClassifier
from main.base.module.classifier import MultiClassifier

class ConfigModel(LightningModule, ConfigObject):
    ''''
    Abstract configurable core model
    '''

    def set_learning(self, loss, optimizer, scheduler=None):
        '''
        Set model learning parameters
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self, batch, batch_index):
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_index):
        return self.step(batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.step(batch, batch_index)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return self.optimizer, self.scheduler

class WrapperModel(ConfigModel):
    '''
    Abstract model wrapping a neural module
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

    def mount_classifier(self, dataset):
        '''
        Mount the classifier by reading dataset labels
        '''
        self.module.set_num_classes(dataset.get_num_classes())

class SingleClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a single classifier
    '''

    def __init__(self, encoder):
        super().__init__(SingleClassifier(encoder))

    def set_num_classes(self, num_classes):
        self.module.set_num_classes(num_classes)

    @classmethod
    def config_args(cls, config, core_nodes):
        encoder = config_core_object1(config.encoder, core_nodes,
                                        MODEL_MODULE)
                                        
        return [ encoder ]

class MultiClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a multiple classifier
    '''

    def __init__(self, encoder):
        super().__init__(MultiClassifier(encoder))
        raise NotImplementedError