from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.config import get_program_config
from aidenv.app.config import search_config_key
from aidenv.app.basic.params import BASE_KEY
from aidenv.app.basic.params import DETERM_KEY
from aidenv.app.basic.params import CORE_KEY
from aidenv.app.basic.config import build_base
from aidenv.app.basic.config import build_determinism
from aidenv.app.basic.config import build_core

# sections parsers

def parse_base(config):
    config = search_config_key(config, BASE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_KEY))

    build_base(config)
    
    print('Base configuration done')

def parse_determinism(config):
    config = search_config_key(config, DETERM_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_KEY))

    build_determinism(config)

    print('Determinism configuration done')

def parse_core(config):
    config = search_config_key(config, CORE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(CORE_KEY))

    task = build_core(config)

    print("Core configuration done")
    return task

# main runners

def run_task(env_name, *args):
    '''
    Entry point for an aidenv core task.
    env_name: name of the environment
    args: program arguments
    '''
    #print(' ---=== AIDENV ===--- \n')
    print(f'{env_name} environment started')

    config = get_program_config()

    parse_base(config)
    parse_determinism(config)
    task = parse_core(config)
    task.run(*args)

    print('')
    print(f'{env_name} environment finished')