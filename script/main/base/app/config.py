from omegaconf import OmegaConf

from main.base.util.string import upper1, upper_identifier
from main.base.app.reflection import get_package_name, get_class

# basic stuff

def load_config(file_path):
    '''
    Load into memory a configuration file.
    config_path: file path
    '''
    return OmegaConf.load(file_path)

class ConfigObject:
    '''
    Configurable core object from file
    '''

    @classmethod
    def config_args(cls, config, core_nodes):
        '''
        Build args for the class from configuration.
        config: configuration object
        core_nodes: nodes of the path of a core location
        '''
        raise NotImplementedError

    @classmethod
    def super_config_args(cls, config, core_nodes, super_index=0):
        '''
        Build args for the inherited class from configuration.
        config: configuration object
        core_nodes: nodes of the path of a core location
        super_index: index of the super class
        '''
        return cls.__bases__[super_index].config_args(config, core_nodes)

    @classmethod
    def from_config(cls, config, core_nodes):
        '''
        Build a class instance from configuration.
        config: configuration object
        core_nodes: nodes of the path of a core location
        '''
        args = cls.config_args(config, core_nodes)
        return cls(*args)

# abstract objects configs

CLASS1_SEPARATOR = '_'

CLASS_NOT_FOUND_MSG = lambda clsname: "class {clsname} not found"
FAILED_CONFIG_MSG = lambda clsname: "failed to configure {clsname} object"

def config_core_prompt(config):
    '''
    Load core location from a configuration.
    config: configuration object
    '''
    return config.core.prompt

def config_object(class_constr, core_nodes, module_name, class_name):
    '''
    Abstract builder of object from a configuration field.
    class_constr: constructor for the class
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    package_name = get_package_name(core_nodes)
    cls = get_class(package_name, module_name, class_name)

    if cls is None:
        raise ValueError(CLASS_NOT_FOUND_MSG(class_name))

    obj = class_constr(cls)
    if obj is None:
        raise RuntimeError(FAILED_CONFIG_MSG(class_name))

    return obj

def config_object1(class_constr, core_nodes, module_name, class_name):
    '''
    Abstract builder of an object with passed class name.
    class_constr: constructor for the class
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    return config_object(class_constr, core_nodes,
                        module_name, class_name)

def config_object2(class_constr, core_nodes, module_name, class_name):
    '''
    Abstract builder of an object with class name equal to configuration field.
    class_constr: constructor for the class
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    return config_object(class_constr, core_nodes,
                        module_name, upper_identifier(class_name,
                                                    CLASS1_SEPARATOR))

def config_object3(class_constr, core_nodes, module_name, config):
    '''
    Abstract builder of an object with class name inside configuration field.
    class_constr: constructor for the class
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    config: configuration object
    '''
    return config_object(class_constr, core_nodes,
                        module_name, config.name)

# main objects configs

def config_simple_object1(config, core_nodes, module_name, class_name, args=[]):
    '''
    Configure a simple1 object.
    Example for ClassName:
    class_name:
        constr_arg0: ...
        constr_arg1: ...
    config: configuration object
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    args: additional args passed to constructor
    '''
    return config_object2(lambda cls: cls(*args, **config[class_name]),
                            core_nodes[:-1], module_name, class_name)

def config_simple_object2(config, core_nodes, module_name):
    '''
    Configure a simple2 object.
    Example for ClassName:
    {field}:
        name: ClassName
        params:
            constr_arg0: ...
            constr_arg0: ...
    config: configuration object
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    '''
    params = config.params if not config.params is None else {}
    return config_object1(lambda cls: cls(**params),
                            core_nodes[:-1], module_name, config.name)

def config_core_object1(config, core_nodes, module_name):
    '''
    config: configuration object
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    '''
    return config_object3(lambda cls: cls.from_config(config.params, core_nodes),
                            core_nodes, module_name, config)

def config_core_object2(config, core_nodes, module_name, core_suffix=True):
    '''
    config: configuration object
    core_nodes: nodes of the path of a core location
    module_name: name of the module file inside the core location
    '''
    class_prefix = upper1(core_nodes[-1])
    if core_suffix:
        class_suffix = upper1(core_nodes[-2])
    else:
        class_suffix = config.name
    class_name = f"{class_prefix}{class_suffix}"
    return config_object(lambda cls: cls.from_config(config.params, core_nodes),
                            core_nodes[:-1], module_name, class_name)