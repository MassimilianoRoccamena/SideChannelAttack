import torch.nn as nn

class Classifier:
    '''
    Abstract classifier object
    '''

    def mount_labels(self, labels):
        '''
        Mount the classification labelling.
        labels: classification labels
        '''
        self.labels = labels

class SingleClassifier(nn.Module, Classifier):
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
        self.mount_labels(labels)

    def mount_labels(self, labels):
        super().mount_labels(labels)
        if labels is None:
            self.softmax = None
        else:
            self.softmax = nn.Softmax(len(labels))

    def forward(self, x):
        encoding = self.encoder(x)
        prediction = self.sofmax(encoding)
        return prediction

class MultiClassifier(nn.Module):
    '''
    Multiple softmax classification module
    '''

    def __init__(self, encoder, num_classes=None):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError