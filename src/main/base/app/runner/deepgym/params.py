import os

from main.base.app.params import CONFIG_DIR

TENSORBOARD_DIR = 'tensorboard'
EXECUTABLE_CONF_PATH = os.path.join(CONFIG_DIR, 'deep-gym.yaml')

NEPTUNE_PRJ_NAME = 'SideChannelAttack'
NEPTUNE_USER_ENV = 'NEPTUNE_API_USER'
NEPTUNE_TOKEN_ENV = 'NEPTUNE_API_TOKEN'