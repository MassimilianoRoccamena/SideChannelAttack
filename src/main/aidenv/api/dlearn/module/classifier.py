import torch.nn as nn

from aidenv.api.dlearn.module.config import CoreModule

class Classifier:
    '''
    Abstract classifier object
    '''

    def set_labels(self, labels):
        '''
        Set the classification labels.
        labels: classification classes
        '''
        self.labels = labels

class SingleClassifier(CoreModule, Classifier):
    '''
    Single softmax classification module
    '''

    def __init__(self, encoder, labels=None):
        '''
        Create new single classifier.
        encoder: encoder module
        labels: classification labels
        '''
        super().__init__()
        self.encoder = encoder
        self.scoring = None
        self.softmax = nn.Softmax()
        self.set_labels(labels)

    def set_input_shape(self, input_shape):
        self.encoder.set_input_shape(input_shape)
        super().set_input_shape(input_shape)

    def set_labels(self, labels):
        super().set_labels(labels)
        if labels is None:
            self.scoring = None
        else:
            self.scoring = nn.Linear(self.encoder.encoding_size,
                                        len(labels))

    def forward(self, x):
        y = self.encoder(x)
        y = self.scoring(y)
        y = self.softmax(y)
        return y

class MultiClassifier(CoreModule, Classifier):
    '''
    Multiple softmax classification module
    '''

    def __init__(self, encoder, labels=None):
        '''
        Create new multiple classifier.
        encoder: encoder module
        labels:  classification labels
        '''
        super().__init__()
        self.encoder = encoder
        self.set_labels(labels)
        raise NotImplementedError

    def set_labels(self, labels):
        super().set_labels(labels)
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError