from torch.utils.data import Dataset

from main.base.app.config import ConfigObject

class ConfigDataset(ConfigObject, Dataset):
    ''''
    Abstract configurable dataset
    '''

    pass