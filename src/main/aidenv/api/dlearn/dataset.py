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

    def current_label(self, *args):
        '''
        Current batch labels, by global passed batch args.
        '''
        raise NotImplementedError

    def all_labels(self):
        '''
        Labels of the classification.
        '''
        raise NotImplementedError

class ClassificationDataset(DeepDataset, Classification):
    ''''
    Abstract classification dataset.
    '''

    pass