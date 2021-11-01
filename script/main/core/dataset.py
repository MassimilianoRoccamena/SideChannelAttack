from torch.utils.data import Dataset

from main.base.launcher.config import ConfigParseable

class ConfigDataset(ConfigParseable, Dataset):
    ''''
    Abstract configurable core dataset
    '''

    pass