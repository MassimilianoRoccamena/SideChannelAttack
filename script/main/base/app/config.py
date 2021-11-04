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

class CoreObject:
    '''
    Core object, configurable from file
    '''

    @classmethod
    def build_args(cls, config, core_prompt):
        '''
        Build args for a class from configuration.
        config: configuration object
        core_prompt: nodes of the path of a core location
        '''
        raise NotImplementedError

    @classmethod
    def build_super_args(cls, config, core_prompt, super_index=0):
        '''
        Build args of a parent class from configuration.
        config: configuration object
        core_prompt: nodes of the path of a core location
        super_index: index of the super class
        '''
        return cls.__bases__[super_index].build_args(config, core_prompt)

    @classmethod
    def from_config(cls, config, core_prompt):
        '''
        Build a class instance from configuration.
        config: configuration object
        core_nodes: nodes of the path of a core location
        '''
        args = cls.build_args(config, core_prompt)
        return cls(*args)

# abstract objects configs

CLASS1_SEPARATOR = '_'

CLASS_NOT_FOUND_MSG = lambda clsname: "class {clsname} not found"
FAILED_CONFIG_MSG = lambda clsname: "failed to configure {clsname} object"

def build_object(class_constr, core_prompt, module_name, class_name):
    '''
    Abstract builder of objects from a module class.
    class_constr: constructor for the class
    core_prompt: nodes of the path of a core location
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    package_name = get_package_name(core_prompt)
    cls = get_class(package_name, module_name, class_name)

    if cls is None:
        raise ValueError(CLASS_NOT_FOUND_MSG(class_name))

    obj = class_constr(cls)
    if obj is None:
        raise RuntimeError(FAILED_CONFIG_MSG(class_name))

    return obj

def build_object1(class_constr, core_prompt, module_name, class_name):
    '''
    Abstract builder of an object with passed class name.
    class_constr: constructor for the class
    core_prompt: nodes of the path of a core location
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    return build_object(class_constr, core_prompt,
                        module_name, class_name)

def build_object2(class_constr, core_prompt, module_name, class_name):
    '''
    Abstract builder of an object with class name equal to configuration field.
    class_constr: constructor for the class
    core_prompt: nodes of the path of a core location
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    return build_object(class_constr, core_prompt,
                        module_name, upper_identifier(class_name,
                                                    CLASS1_SEPARATOR))

# simple objects

def build_simple_object1(config, core_prompt, module_name, class_name, args=[], kwargs={}):
    '''
    Build a collapsed constructor based object.
    Example for ClassName:
    class_name:
        constr_arg0: ...
        constr_arg1: ...
    config: configuration object
    core_prompt: nodes of the path of a core location
    module_name: name of the module file inside the core location
    args: additional args passed to constructor
    '''
    params = config[class_name] if not config[class_name] is None else {}
    params = dict(params)
    params.update(kwargs)
    return build_object2(lambda cls: cls(*args, **params),
                            core_prompt[:-1], module_name, class_name)

def build_simple_object2(config, core_prompt, module_name, args=[], kwargs={}):
    '''
    Build an expanded constructor based object.
    Example for ClassName:
    {field}:
        name: ClassName
        params:
            constr_arg0: ...
            constr_arg0: ...
    config: configuration object
    core_prompt: nodes of the path of a core location
    module_name: name of the module file inside the core location
    '''
    params = config.params if not config.params is None else {}
    params = dict(params)
    params.update(kwargs)
    return build_object1(lambda cls: cls(*args, **params),
                            core_prompt[:-1], module_name, config.name)

# core objects

def build_core_prompt(config):
    '''
    Build core prompt from a configuration.
    config: configuration object
    '''
    return config.core.prompt

def build_core_object1(config, core_prompt, module_name):
    '''
    Build an expanded core object.
    Example for ClassName:
    {field}:
        name: ClassName
        params:
            constr_arg0: ...
            constr_arg0: ...
    config: configuration object
    core_prompt: nodes of the path of a core location
    module_name: name of the module file inside the core location
    '''
    params = config.params if not config.params is None else {}
    return build_object1(lambda cls: cls.from_config(params, core_prompt),
                            core_prompt[:-1], module_name, config.name)

def build_core_object2(config, core_prompt, module_name, core_suffix=True):
    '''
    Build an expanded core object.
    This object exploits core prompt for locating the class name.
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
    class_prefix = upper1(core_prompt[-1])
    if core_suffix:
        class_suffix = upper1(core_prompt[-2])
    else:
        class_suffix = config.name
    class_name = f"{class_prefix}{class_suffix}"

    params = config.params if not config.params is None else {}
    return build_object1(lambda cls: cls.from_config(params, core_prompt),
                            core_prompt[:-1], module_name, class_name)