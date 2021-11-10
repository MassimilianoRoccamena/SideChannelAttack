from utils.reflection import get_package_name

# errors

CONFIG_NOT_FOUND_MSG = lambda id: f'{id} configuration field not found'
ENV_NOT_FOUND_MSG = lambda env: f'{env} environmental variable not found'

# language lex

CLASS_NAME_KEY = 'name'
CLASS_PARAMS_KEY = 'params'

# env vars

AIDENV_CONFIG_ENV = 'AIDENV_CONFIG'

# basic filesystem

LOG_DIR = ".log"

DATASET_MODULE = 'dataset'
MODEL_MODULE = 'model'
LEARNING_MODULE = 'learning'

# core filesystem

CORE_PACKAGE = None

def set_core_package(origin, append_core=True):
    '''
    Set core package used by the configuration system.
    origin: nodes of the path to the core package
    append_core: wheter to append a core package at the end
    '''
    if append_core:
        origin = origin + ['core']

    global CORE_PACKAGE
    CORE_PACKAGE = get_package_name(origin)

def get_core_package():
    '''
    Get core package used by the configuration system.
    '''
    return CORE_PACKAGE