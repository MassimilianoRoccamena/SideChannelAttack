from torch.utils.data import Dataset as TorchDataset

from aidenv.api.dataset import CoreDataset

# basic

class DeepDataset(TorchDataset, CoreDataset):
    ''''
    Abstract deep learning dataset.
    '''

    def data_shape(self):
        '''
        Returns shape of each data point
        '''
        raise NotImplementedError

# classification

class Classification:
    ''''
    Abstract classification interface.
    '''

    def all_labels(self):
        '''
        Labels of the classification.
        '''
        raise NotImplementedError

    def current_label(self):
        '''
        Label of the current sample
        '''
        raise NotImplementedError

class ClassificationDataset(DeepDataset, Classification):
    ''''
    Abstract classification dataset.
    '''

    pass