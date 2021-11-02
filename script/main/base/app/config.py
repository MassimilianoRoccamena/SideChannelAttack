from omegaconf import OmegaConf

from main.base.util.string import upper_identifier
from main.base.app.reflection import get_package_name, get_class

# basic stuff

def load_config(config_path):
    return OmegaConf.load(config_path)

class ConfigObject:
    '''
    Configurable core object from .yaml file
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

# configuring object

def config_core_prompt(config):
    return config.core.prompt

CLASS_NOT_FOUND_MSG = lambda clsname: "class {clsname} not found"
FAILED_CONFIG_MSG = lambda clsname: "failed to configure {clsname} object"

def config_object(class_constr, core_nodes, module_name, class_name):
    package_name = get_package_name(core_nodes)
    cls = get_class(package_name, module_name, class_name)

    if cls is None:
        raise ValueError(CLASS_NOT_FOUND_MSG(class_name))

    obj = class_constr(cls)
    if obj is None:
        raise RuntimeError(FAILED_CONFIG_MSG(class_name))

    return obj

def config_object1(object_constr, core_nodes, module_name, config, field_name):
    return config_object(object_constr, core_nodes,
                        module_name, upper_identifier(field_name, '_'))

def config_object2(object_constr, core_nodes, module_name, field_name):
    return config_object(object_constr, core_nodes,
                        module_name, field_name)

def config_object3(object_constr, core_nodes, module_name, config):
    return config_object(object_constr, core_nodes,
                        module_name, config.name)