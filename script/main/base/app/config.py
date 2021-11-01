from omegaconf import OmegaConf

from main.base.util.string import upper1
from main.base.app.reflection import get_package_name, get_class
from main.base.app.params import TRAINING_CONF_PATH

def load_config(config_path):
    return OmegaConf.load(config_path)

def load_training_config():
    return load_config(TRAINING_CONF_PATH)


class ConfigObject:
    '''
    Configurable object from .yaml file
    '''

    @classmethod
    def config_args(cls, config, core_nodes):
        raise NotImplementedError

    @classmethod
    def super_config_args(cls, config, core_nodes, super_index=0):
        return cls.__bases__[super_index].config_args(config, core_nodes)

    @classmethod
    def from_config(cls, config, core_nodes):
        args = cls.config_args(config, core_nodes)
        return cls(*args)


def config_core(config):
    return config.core

CLASS_NOT_FOUND_MSG = lambda clsname: "class {clsname} not found"
FAILED_CONFIG_MSG = lambda clsname: "failed to configure {clsname} object"

def config_object(config, core_nodes, module_name):
    package_name = get_package_name(core_nodes)
    class_name = config.name
    cls = get_class(package_name, module_name, class_name)

    if cls is None:
        raise ValueError(CLASS_NOT_FOUND_MSG(class_name))

    obj = cls.from_config(config.params, core_nodes)
    if obj is None:
        raise RuntimeError(FAILED_CONFIG_MSG(class_name))

    return obj

def config_core_object(config, core_nodes, module_name, core_suffix=True):
    package_name = get_package_name(core_nodes[:-1])
    class_prefix = upper1(core_nodes[-1])
    if core_suffix:
        class_suffix = upper1(core_nodes[-2])
    else:
        class_suffix = config.name
    class_name = f"{class_prefix}{class_suffix}"
    cls = get_class(package_name, module_name, class_name)

    if cls is None:
        raise ValueError(CLASS_NOT_FOUND_MSG(class_name))

    obj = cls.from_config(config.params, core_nodes)
    if obj is None:
        raise RuntimeError(FAILED_CONFIG_MSG(class_name))

    return obj