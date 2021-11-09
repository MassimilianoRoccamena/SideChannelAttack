from omegaconf import OmegaConf

from utils.string import upper1, upper_identifier
from aidenv.app.reflection import get_core_package_name
from aidenv.app.reflection import get_core_class

# basic stuff

def load_config(file_path):
    '''
    Load into memory a configuration file.
    config_path: file path
    '''
    return OmegaConf.load(file_path)

class CoreObject:
    '''
    Core object, configurable from file.
    '''

    @classmethod
    def build_kwargs(cls, config, prompt):
        '''
        Build kwargs for a class from configuration.
        config: configuration object or dict
        prompt: nodes of the path from the core package
        '''
        if type(config) is dict:
            return config
        else:
            return dict(config)

    @classmethod
    def update_kwargs(cls, config, **kwargs):
        '''
        Update kwargs for a class constructor.
        config: configuration object
        prompt: nodes of the path from the core package
        '''
        config = dict(config)
        for k,v in kwargs.items():
            config[k] = v
        return config

    @classmethod
    def build_super_kwargs(cls, config, prompt, super_index=0):
        '''
        Build args of a parent class from configuration.
        config: configuration object
        prompt: nodes of the path from the core package
        super_index: index of the super class
        '''
        return cls.__bases__[super_index].build_kwargs(config, prompt)

    @classmethod
    def from_config(cls, config, prompt):
        '''
        Build a class instance from configuration by calling
        constructor with kwargs.
        config: configuration object
        prompt: nodes of the path from the core package
        '''
        kwargs = cls.build_kwargs(config, prompt)
        return cls(**kwargs)

# generic objects builders

CLASS_NOT_FOUND_MSG = lambda clsname: "class {clsname} not found"
FAILED_CONFIG_MSG = lambda clsname: "failed to configure {clsname} object"

def build_object(class_constr, prompt, module_name, class_name):
    '''
    Abstract builder of objects from a module class.
    class_constr: constructor for the class
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    package_name = get_core_package_name(prompt)
    cls = get_core_class(package_name, module_name, class_name)

    if cls is None:
        raise TypeError(CLASS_NOT_FOUND_MSG(class_name))

    obj = class_constr(cls)
    if obj is None:
        raise RuntimeError(FAILED_CONFIG_MSG(class_name))

    return obj

def build_object1(class_constr, prompt, module_name, class_name):
    '''
    Abstract builder of an object with passed class name.
    class_constr: constructor for the class
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    return build_object(class_constr, prompt,
                        module_name, class_name)

def build_object2(class_constr, prompt, module_name, class_name):
    '''
    Abstract builder of an object with class name equal to configuration field.
    class_constr: constructor for the class
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    return build_object(class_constr, prompt,
                        module_name, upper_identifier(class_name,
                                                    '_'))

# simple objects builders

def build_simple_object1(config, prompt, module_name, class_name, args, kwargs):
    '''
    Build a collapsed constructor based object.
    Example for ClassName:
    class_name:
        constr_arg0: ...
        constr_arg1: ...
    config: configuration object
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    class_name: name of the class
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    '''
    params = config[class_name] if not config[class_name] is None else {}
    params = dict(params)
    params.update(kwargs)
    return build_object2(lambda cls: cls(*args, **params),
                            prompt[:-1], module_name, class_name)

def build_simple_object2(config, prompt, module_name, args, kwargs):
    '''
    Build an expanded constructor based object.
    Example for ClassName:
    {field}:
        name: ClassName
        params:
            constr_arg0: ...
            constr_arg0: ...
    config: configuration object
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    '''
    params = config.params if not config.params is None else {}
    params = dict(params)
    params.update(kwargs)
    return build_object1(lambda cls: cls(*args, **params),
                            prompt[:-1], module_name, config.name)

# core objects builders

def build_core_object1(config, prompt, module_name):
    '''
    Build an expanded core object.
    Example for ClassName:
    {field}:
        name: ClassName
        params:
            constr_arg0: ...
            constr_arg0: ...
    config: configuration object
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    '''
    params = config.params if not config.params is None else {}
    return build_object1(lambda cls: cls.from_config(params, prompt),
                            prompt[:-1], module_name, config.name)

def build_core_object2(config, prompt, module_name, prompt_suffix):
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
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    prompt_duffix: if true append part of prompt to class name
    '''
    class_prefix = upper1(prompt[-1])
    if prompt_suffix:
        class_suffix = upper1(prompt[-2])
    else:
        class_suffix = config.name
    class_name = f"{class_prefix}{class_suffix}"

    params = config.params if not config.params is None else {}
    return build_object1(lambda cls: cls.from_config(params, prompt),
                            prompt[:-1], module_name, class_name)