from aidenv.api.dlearn.config import build_model_kwarg
from aidenv.api.dlearn.model.wrapper import WrapperModel
from aidenv.api.dlearn.module.classifier import SingleClassifier, SingleClassifierAdvanced
from aidenv.api.dlearn.module.classifier import MultiClassifier

class ClassifierModel(WrapperModel):
    '''
    Abstract model wrapping a classifier.
    '''

    def __init__(self, classifier):
        '''
        Create new classifier model.
        classifier: classifier module
        '''
        super().__init__(classifier)

    @classmethod
    @build_model_kwarg('encoder')
    def build_kwargs(cls, config):
        pass

    def set_labels(self, labels):
        '''
        Set the classification labels.
        labels: classification classes
        '''
        self.labels = labels

    def mount(self, *args, **kwargs):
        dataset = args[0]
        labels = dataset.all_labels()
        super().mount(*args, labels=labels)
        self.labels = labels
        self.module.set_labels(labels)

    def step_batch(self, batch):
        loss, target, prediction = super().step_batch(batch)
        encoding = self.module.encoder.input_encoded
        return loss, target, prediction, encoding

    def compute_step(self, batch, prefix):
        step_results = self.step_batch(batch)
        outputs = {'loss':step_results[0],'target':step_results[1],
                    'prediction':step_results[2],'encoding':step_results[3]}
        return outputs

class SingleClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a single classifier
    '''

    def __init__(self, encoder):
        super().__init__(SingleClassifier(encoder))

class SingleClassifierAdvancedModel(ClassifierModel):
    '''
    Abstract model wrapping a single classifier
    '''

    def __init__(self, encoder, layers):
        super().__init__(SingleClassifierAdvanced(encoder, layers))

class MultiClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a multiple classifier
    '''

    def __init__(self, encoder, loggables=None):
        super().__init__(MultiClassifier(encoder))
        raise NotImplementedError