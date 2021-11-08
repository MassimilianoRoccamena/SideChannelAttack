import os

from main.mlenv.app.params import CONFIG_DIR

EXECUTABLE_CONF_PATH = os.path.join(CONFIG_DIR, 'deep-gym.yaml')
TENSORBOARD_DIR = 'tensorboard'
CHECKPOINT_DIR = 'checkpoints'

NEPTUNE_PRJ_NAME = 'SideChannelAttack'
NEPTUNE_USER_ENV = 'NEPTUNE_API_USER'
NEPTUNE_TOKEN_ENV = 'NEPTUNE_API_TOKEN'