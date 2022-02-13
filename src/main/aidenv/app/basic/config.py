import os
from datetime import datetime
import numpy as np

from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.params import TASK_MODULE
from aidenv.app.config import get_program_output_dir
from aidenv.app.config import get_program_config
from aidenv.app.config import search_config_key
from aidenv.app.config import set_program_base
from aidenv.app.config import build_object_expanded
from aidenv.app.logging import log_program
from aidenv.app.basic.params import *

# objects

def build_task_object(config, args=None, kwargs=None):
    '''
    Build an expanded task object.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_object_expanded(config, TASK_MODULE, args, kwargs, core_obj=True)
 
# kwarg

def build_task_kwarg(kwarg_name):
    '''
    Decorator for auto-build of a task object kwarg
    kwarg_name: parameter name
    '''
    def build_kwarg(cls, config, param):
        obj = build_task_object(config[param])
        kwargs = {param:obj}
        config = cls.update_kwargs(config, **kwargs)
        return config

    def decorator(f):
        def wrapper(cls, config):
            return build_kwarg(cls, config, kwarg_name)
        return wrapper
    
    return decorator

# base

INVALID_ORIGIN_MSG = 'program origin contains no path'

def build_base(config):
    # core origin
    origin = search_config_key(config, BASE_ORIGIN_KEY)
    if origin is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_ORIGIN_KEY))
    if len(origin) == 0:
        raise KeyError(INVALID_ORIGIN_MSG)

    # name
    name = search_config_key(config, BASE_NAME_KEY)
    if name is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_NAME_KEY))

    # id
    id = search_config_key(config, BASE_ID_KEY)
    if id is None:
        id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        id = str(id)

    # log dir
    out_dir = get_program_output_dir()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    name_dir = os.path.join(out_dir, name)
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)

    id_dir = os.path.join(name_dir, id)
    if os.path.exists(id_dir):
        raise RuntimeError(f'experiment id {id} for experiment name {name} already exists')
    else:
        os.mkdir(id_dir)

    print(f'Log directory path is {id_dir}')
    log_program(get_program_config(), id_dir)
    print('Saved program configuration')

    # description
    descr = search_config_key(config, BASE_DESCR_KEY)
    if descr is None:
        descr = 'no description provided'

    origin = list(origin)
    set_program_base(origin, name, id, id_dir, descr)

# determinism

ADD_DETERMINISM = None
def add_determinism(f):
    global ADD_DETERMINISM
    ADD_DETERMINISM = f

def build_determinism(config):
    seed = search_config_key(config, DETERM_SEED_KEY)
    if seed is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_SEED_KEY))
    np.random.seed(seed=seed)

    if not ADD_DETERMINISM is None:
        ADD_DETERMINISM(config)

# core

def build_core(config):
    task = build_task_object(config)
    print('Loaded core task')

    return task
