from torch.utils.data import Dataset

from main.base.app.config import CoreObject

class CoreDataset(Dataset, CoreObject):
    ''''
    Abstract core dataset
    '''

    pass

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