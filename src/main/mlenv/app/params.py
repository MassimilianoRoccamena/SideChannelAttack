from main.utils.reflection import get_package_name

# basic

CONFIG_DIR = 'config'
LOG_DIR = ".log"

DATASET_MODULE = 'dataset'
MODEL_MODULE = 'model'
LEARNING_MODULE = 'learning'

# core

CORE_PACKAGE = None

def set_core_package(origin, append_main=True, append_core=True):
    '''
    Set core package used by the configuration system.
    origin: nodes of the path to the core package
    append_main: wheter to append main package as first node
    '''
    if append_main:
        origin = ['main'] + origin
    if append_core:
        origin = origin + ['core']

    global CORE_PACKAGE
    CORE_PACKAGE = get_package_name(origin)

def get_core_package():
    '''
    Get core package used by the configuration system.
    '''
    return CORE_PACKAGE