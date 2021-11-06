from torch.utils.data import Dataset

from main.base.app.config import CoreObject

class CoreDataset(Dataset, CoreObject):
    ''''
    Abstract core dataset
    '''

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

# classification

class ClassificationDataset(CoreDataset):
    ''''
    Abstract classification dataset
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