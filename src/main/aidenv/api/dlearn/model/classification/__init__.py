from aidenv.api.dlearn.config import build_model_kwarg
from aidenv.api.dlearn.model.wrapper import WrapperModel
from aidenv.api.dlearn.model.classification.logging import LoggableAccuracy
from aidenv.api.dlearn.model.classification.logging import LoggableConfusionMatrix
from aidenv.api.dlearn.module.classifier import SingleClassifier
from aidenv.api.dlearn.module.classifier import MultiClassifier

class ClassifierModel(WrapperModel):
    '''
    Abstract model wrapping a classifier.
    '''

    def __init__(self, encoder):
        '''
        Create new classifier model.
        encoder: encoder module
        '''
        super().__init__(encoder)

    @classmethod
    @build_model_kwarg('encoder')
    def build_kwargs(cls, config, prompt):
        pass

    def mount(self, *args, **kwargs):
        dataset = args[0]
        labels = dataset.all_labels()

        self.module.set_labels(labels)

        class_logs = { 'accuracy' : LoggableAccuracy(),
                       'confusion' : LoggableConfusionMatrix(len(labels)) }
        self.add_loggables(class_logs, 'train')
        self.add_loggables(class_logs, 'valid')
        self.add_loggables(class_logs, 'test')

        super().mount(dataset)

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

    def __init__(self, encoder, loggables=None):
        super().__init__(MultiClassifier(encoder))
        raise NotImplementedError