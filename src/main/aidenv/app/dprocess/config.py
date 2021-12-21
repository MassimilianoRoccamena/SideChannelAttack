import os
from datetime import datetime
import numpy as np

from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.params import TASK_MODULE
from aidenv.app.params import set_core_package
from aidenv.app.config import get_program_output_dir
from aidenv.app.config import get_program_config
from aidenv.app.config import search_config_key
from aidenv.app.config import build_core_object1
from aidenv.app.config import build_core_object2
from aidenv.app.logging import log_program
from aidenv.app.dprocess.params import *

# executable objects builders

def build_task_object1(config, prompt):
    '''
    Build an expanded task object.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object1(config, prompt, TASK_MODULE)

def build_task_object2(config, prompt, kwargs={}):
    '''
    Build an expanded task object.
    This object exploits core prompt for locating the class name.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object2(config, prompt, TASK_MODULE, True,
                                kwargs)

# base builders

INVALID_PROMPT_MSG = 'selected prompt is not valid'

def build_base(config):
    # core origin
    origin = search_config_key(config, BASE_ORIGIN_KEY)
    if origin is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_ORIGIN_KEY))

    set_core_package(list(origin))

    # core prompt
    prompt = search_config_key(config, BASE_PROMPT_KEY)
    if prompt is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_PROMPT_KEY))

    # name
    name = search_config_key(config, BASE_NAME_KEY)
    if name is None:
        p = prompt
        if len(prompt) == 3:
            name = f"{p[0]}{p[1]}{p[2]}"
        elif len(prompt) == 2:
            name = f"{p[0]}{p[1]}"
        else:
            raise ValueError(INVALID_PROMPT_MSG)
        print('Using default experiment name')

    # id and log dir
    id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = get_program_output_dir()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    env_dir = os.path.join(out_dir, 'dprocess')
    if not os.path.exists(env_dir):
        os.mkdir(env_dir)

    name_dir = os.path.join(env_dir, name)
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)

    log_dir = os.path.join(name_dir, id)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    print(f'Log directory path is {log_dir}')
    log_program(get_program_config(), log_dir)
    print('Saved program configuration')

    # description
    descr = search_config_key(config, BASE_DESCR_KEY)
    if prompt is None:
        descr = ''

    return origin[-1], prompt, name, id, log_dir, descr

# determinism builders

def build_determinism(config):
    seed = search_config_key(config, DETERM_SEED_KEY)
    if seed is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_SEED_KEY))

    np.random.seed(seed=seed)

# core builders

def build_task(config, prompt, log_dir):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG('dataset'))

    task = build_task_object2(config, prompt,
                                    kwargs={'log_dir':log_dir})
    
    return task

def build_core(config, prompt, log_dir):
    task = search_config_key(config, CORE_TASK_KEY)
    task = build_task(task, prompt, log_dir)
    print('Loaded task')

    return task
