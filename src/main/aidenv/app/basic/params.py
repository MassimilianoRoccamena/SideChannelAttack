# environment parameters

ENVIRONMENT_NAME = None

def set_environment_name(env_name):
    global ENVIRONMENT_NAME
    ENVIRONMENT_NAME = env_name

def get_environment_name():
    return ENVIRONMENT_NAME

# language lex

BASE_KEY = 'base'
BASE_ORIGIN_KEY = 'origin'
BASE_NAME_KEY = 'name'
BASE_ID_KEY = 'id'
BASE_DESCR_KEY = 'description'

DETERM_KEY = 'determinism'
DETERM_SEED_KEY = 'seed'

CORE_KEY = 'core'

# env vars

# filesystem
