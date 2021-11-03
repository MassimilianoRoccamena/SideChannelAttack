import torch
import torch.nn as nn

class SingleClassifier(nn.Module):
    '''
    Single softmax classification module
    '''

    def __init__(self, encoder, num_classes=None):
        super().__init__()
        self.encoder = encoder
        self.set_num_classes(num_classes)

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
        if num_classes is None:
            self.softmax = None
        else:
            self.softmax = nn.Softmax(num_classes)

    def forward(self, x):
        encoding = self.encoder(x)
        prediction = self.sofmax(encoding)
        return prediction

class MultiClassifier(nn.Module):
    '''
    Multiple softmax classification module
    '''

    def __init__(self, encoder, num_classes):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError