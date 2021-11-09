from torch.utils.data import Dataset

from aidenv.api.config import CoreObject

class CoreDataset(Dataset, CoreObject):
    ''''
    Abstract core dataset.
    '''

    def __init__(self, reader):
        '''
        Create new core dataset.
        reader: file reader
        '''
        self.reader = reader

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        raise NotImplementedError

# classification

class Classification:
    ''''
    Abstract classification interface
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

class ClassificationDataset(CoreDataset, Classification):
    ''''
    Abstract classification dataset
    '''

    pass