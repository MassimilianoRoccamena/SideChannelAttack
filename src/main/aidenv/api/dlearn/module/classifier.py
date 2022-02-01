import torch.nn as nn

from aidenv.api.dlearn.module.config import CoreModule

class Classifier:
    '''
    Abstract classifier object.
    '''

    def set_labels(self, labels):
        '''
        Set the classification labels.
        labels: classification classes
        '''
        self.labels = labels

class SingleClassifier(CoreModule, Classifier):
    '''
    Single softmax classification module.
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
            self.scoring = nn.Linear(self.encoder.encoding_size, len(labels))

    def forward(self, x):
        y = self.encoder(x)
        y = self.scoring(y)
        y = self.softmax(y)
        return y

class SingleClassifierAdvanced(CoreModule, Classifier):
    '''
    Single fully connected + softmax classification module.
    '''

    def __init__(self, encoder, layers, labels=None):
        '''
        Create new single classifier.
        encoder: encoder module
        layers: list of number of nodes in each layer
        labels: classification labels
        '''
        super().__init__()
        self.encoder = encoder
        self.scoring = None
        self.softmax = nn.Softmax()
        self.set_labels(labels)

        if len(layers) == 0:
            raise ValueError('fully connected network must have at least one layer')
        self.layers = layers

        self.fc_linears = nn.ModuleList()
        self.fc_activations = nn.ModuleList()
        in_channels = encoder.encoding_size
        for layer in layers:
            out_channels = layer
            linear = nn.Linear(in_channels, out_channels)
            self.fc_linears.append(linear)
            activation = nn.SELU()
            self.fc_activations.append(activation)
            in_channels = out_channels
    
    def set_labels(self, labels):
        super().set_labels(labels)
        if labels is None:
            self.scoring = None
        else:
            self.scoring = nn.Linear(self.layers[-1], len(labels))

    def forward(self, x):
        y = self.encoder(x)
        for lin, act in zip(self.fc_linears, self.fc_activations):
            y = lin(y)
            y = act(y)
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