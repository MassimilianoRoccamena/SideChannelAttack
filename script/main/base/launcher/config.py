from omegaconf import OmegaConf

from main.base.launcher.reflection import package_name, get_class
from main.base.launcher.params import TRAINING_CONF_PATH

def load_config(config_path):
    return OmegaConf.load(config_path)

def load_training_config():
    return load_config(TRAINING_CONF_PATH)

class ConfigParseable:
    def __init__(self, parser):
        self.parser = parser

class ConfigParser:
    def __init__(self, parseable_class):
        self.parseable_class = parseable_class

    def constr_args(self, config):
        raise NotImplementedError

    def call_constr(self, config):
        return self.parseable_class(*self.constr_args(config))

    def parse_config(self, config):
        raise self.call_constr(config)