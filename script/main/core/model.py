from torch.nn import Module

from main.base.launcher.config import ConfigParseable

class ConfigModel(ConfigParseable, Module):
    ''''
    Abstract configurable core model
    '''

    pass