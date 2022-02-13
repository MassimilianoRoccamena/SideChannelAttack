import os
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from omegaconf.dictconfig import DictConfig

from utils.string import upper_identifier
from utils.reflection import get_package_name
from utils.reflection import get_module_path
from utils.reflection import get_class
from aidenv.app.params import ENV_NOT_FOUND_MSG
from aidenv.app.params import AIDENV_INPUT_ENV
from aidenv.app.params import AIDENV_OUTPUT_ENV
from aidenv.app.params import AIDENV_PROGRAM_ENV
from aidenv.app.params import CLASS_NAME_KEY
from aidenv.app.params import CLASS_PARAMS_KEY

# aidenv utils

def load_env_var(name):
    '''
    Load an environmental variable redirecting exceptions.
    name: variable name
    '''
    try:
        var = os.environ[name]
    except KeyError:
        raise EnvironmentError(ENV_NOT_FOUND_MSG(name))
    return var

# aidenv program config

PROGRAM_INPUT_DIR = load_env_var(AIDENV_INPUT_ENV)

def get_program_input_dir():
    return PROGRAM_INPUT_DIR

PROGRAM_OUTPUT_DIR = load_env_var(AIDENV_OUTPUT_ENV)

def get_program_output_dir():
    return PROGRAM_OUTPUT_DIR

PROGRAM_CONFIG = OmegaConf.load(load_env_var(AIDENV_PROGRAM_ENV))

def get_program_config():
    '''
    Returns the loaded aidenv program configuration.
    '''
    return PROGRAM_CONFIG

def search_config_key(config, key):
    '''
    Load a configuration field putting None if empty or not present.
    config: configuration object
    key: field name of config
    '''
    try:
        loaded = config[key]
    except ConfigKeyError:
        loaded = None
    return loaded

# aidenv program base

PROGRAM_ORIGIN = None
PROGRAM_NAME = None
PROGRAM_ID = None
PROGRAM_LOG_DIR = None
PROGRAM_DESCRIPTION = None

def set_program_base(origin, name, id, log_dir, descr):
    '''
    Set aidenv program base parameters.
    origin: nodes of the path to the core package
    '''
    global PROGRAM_ORIGIN
    PROGRAM_ORIGIN = origin
    global PROGRAM_NAME
    PROGRAM_NAME = name
    global PROGRAM_ID
    PROGRAM_ID = id
    global PROGRAM_LOG_DIR
    PROGRAM_LOG_DIR = log_dir
    global PROGRAM_DESCRIPTION
    PROGRAM_DESCRIPTION = descr

def get_program_origin():
    '''
    Returns the loaded aidenv program origin.
    This is the core package path.
    '''
    return PROGRAM_ORIGIN
def get_program_name():
    '''
    Returns the loaded aidenv program name.
    '''
    return PROGRAM_NAME
def get_program_id():
    '''
    Returns the loaded aidenv program identifier.
    '''
    return PROGRAM_ID
def get_program_log_dir():
    '''
    Returns the loaded aidenv program log directory.
    '''
    return PROGRAM_LOG_DIR
def get_program_description():
    '''
    Returns the loaded aidenv program description.
    '''
    return PROGRAM_DESCRIPTION

# aidenv program core

class CoreObject:
    '''
    Core aidenv object, configurable from a program file in the core package.
    '''

    @classmethod
    def build_kwargs(cls, config):
        '''
        Build kwargs for a class from configuration.
        config: configuration object or dict
        '''
        return config

    @classmethod
    def update_kwargs(cls, config, **kwargs):
        '''
        Update kwargs for a class constructor.
        config: configuration object
        '''
        for k,v in kwargs.items():
            config[k] = v
        return config

    @classmethod
    def build_super_kwargs(cls, config, super_index=0):
        '''
        Build args of a parent class from configuration.
        config: configuration object
        super_index: index of the super class
        '''
        return cls.__bases__[super_index].build_kwargs(config)

    @classmethod
    def from_config(cls, config):
        '''
        Build a class instance from configuration by calling
        constructor with kwargs.
        config: configuration object
        '''
        kwargs = cls.build_kwargs(config)
        return cls(**kwargs)

# aidenv objects builders

def get_core_class(module_name, class_name):
    '''
    Reflective load of a class from core package. Uses origin
    configuration as path to core package.
    module_name: name of the module inside the core package
    class_name: name of the class inside the module
    '''
    package_name = get_package_name(PROGRAM_ORIGIN)
    module_path = get_module_path(package_name, module_name)
    return get_class(module_path, class_name)

NO_CLASS_NAME_MSG = "class name not specified"
FAILED_CONFIG_MSG = lambda cls: "failed to configure {cls} object"

def build_object(class_constr, module_name, class_name):
    '''
    Abstract builder of objects from a module class.
    class_constr: constructor for the class
    module_name: name of the module file inside the core location
    class_name: name of the class inside the module
    '''
    cls = get_core_class(module_name, class_name)

    obj = class_constr(cls)
    if obj is None:
        raise RuntimeError(FAILED_CONFIG_MSG(class_name))

    return obj

def build_object_collapsed(config, module_name, class_name, args, kwargs, core_obj=False):
    '''
    Build a collapsed constructor based object.
    Example for ClassName:
    class_name:
        constr_arg0: ...
        constr_arg1: ...
    config: configuration object
    module_name: name of the module file inside the core package
    class_name: name of the class
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    core_obj: if the object is a core object
    '''
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    params = search_config_key(config, class_name)
    if params is None:
        params = {}
    elif type(params) is DictConfig:
        params = OmegaConf.to_object(params)
    params.update(kwargs)

    if core_obj:
        class_constr = lambda cls: cls.from_config(params)
    else:
        class_constr = lambda cls: cls(*args, **params)

    class_name = upper_identifier(class_name)
    return build_object(class_constr, module_name, class_name)

def build_object_expanded(config, module_name, args, kwargs, core_obj=False):
    '''
    Build an expanded constructor based object.
    Example for ClassName:
    {parent}:
        name: ClassName
        params:
            constr_arg0: ...
            constr_arg0: ...
    config: configuration object
    module_name: name of the module file inside the core package
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    core_obj: if the object is a core object
    '''
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
        
    class_name = search_config_key(config, CLASS_NAME_KEY)
    if class_name is None:
        raise KeyError(NO_CLASS_NAME_MSG)

    params = search_config_key(config, CLASS_PARAMS_KEY)
    if params is None:
        params = {}
    elif type(params) is DictConfig:
        params = OmegaConf.to_object(params)
    params.update(kwargs)

    if core_obj:
        class_constr = lambda cls: cls.from_config(params)
    else:
        class_constr = lambda cls: cls(*args, **params)

    return build_object(class_constr, module_name, class_name)
