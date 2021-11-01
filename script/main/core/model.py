from torch.nn import Module

from main.base.app.config import ConfigObject

class ConfigModel(ConfigObject, Module):
    ''''
    Abstract configurable model
    '''

    pass