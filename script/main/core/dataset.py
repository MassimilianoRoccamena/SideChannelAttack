from torch.utils.data import Dataset

from main.base.app.config import ConfigObject

class ConfigDataset(Dataset, ConfigObject):
    ''''
    Abstract configurable core dataset
    '''

    pass

# classification

class ClassificationDataset(Dataset, ConfigObject):
    ''''
    Abstract classification dataset
    '''

    def get_num_classes():
        raise NotImplementedError