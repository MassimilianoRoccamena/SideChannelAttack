from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.config import get_program_config
from aidenv.app.config import search_config_key
from aidenv.app.dprocess.params import BASE_KEY
from aidenv.app.dprocess.params import DETERM_KEY
from aidenv.app.dprocess.params import CORE_KEY
from aidenv.app.dprocess.config import build_base
from aidenv.app.dprocess.config import build_determinism
from aidenv.app.dprocess.config import build_core

# sections parsers

def parse_base(config):
    config = search_config_key(config, BASE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_KEY))

    origin, prompt, name, id, log_dir, descr = build_base(config)
    
    print('Base configuration done')
    return origin, prompt, name, id, log_dir, descr

def parse_determinism(config):
    config = search_config_key(config, DETERM_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_KEY))

    build_determinism(config)

    print('Determinism configuration done')

def parse_core(config, prompt, log_dir):
    config = search_config_key(config, CORE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(CORE_KEY))

    task = build_core(config, prompt, log_dir)

    print("Core configuration done")
    return task

# main runners

def run(*args):
    '''
    Entry point for dprocess environment
    args: program arguments
    '''
    print(' ---=== AIDENV ===--- \n')
    print("Data processing environment started")
    config = get_program_config()
    hparams = {}

    origin, prompt, name, id, log_dir, descr = parse_base(config)
    parse_determinism(config)
    task = parse_core(config, prompt, log_dir)

    task.run()
    print('')
    print('Data processing environment finished')