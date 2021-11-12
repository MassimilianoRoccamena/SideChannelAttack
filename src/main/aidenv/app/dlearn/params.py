# language lex

BASE_KEY = 'base'
BASE_ORIGIN_KEY = 'origin'
BASE_PROMPT_KEY = 'prompt'
BASE_NAME_KEY = 'name'

DETERM_KEY = 'determinism'
DETERM_SEED_KEY = 'seed'
DETERM_WORKERS_KEY = 'seed_workers'
DETERM_FORCE_KEY = 'force'

LOG_KEY = 'logging'
LOG_TB_KEY = 'tensorboard'
LOG_NEPT_KEY = 'neptune'
LOG_ENABLE_KEY = 'enable'
LOG_NEPT_OFF_KEY = 'offline_mode'

CORE_KEY = 'core'
CORE_DATASET_KEY = 'dataset'
CORE_DATASET_NSAMP_KEY = 'num_samples'
CORE_MODEL_KEY = 'model'
CORE_MODEL_CKPT_KEY = 'checkpoint'

LEARN_KEY = 'learning'
LEARN_SKIP_KEY = 'skip'
LEARN_SPLIT_KEY = 'split'
LEARN_TRAIN_KEY = 'training'
LEARN_VALID_KEY = 'validation'
LEARN_TEST_KEY = 'test'
LEARN_DATA_LOAD_KEY = 'data_loader'
LEARN_EARLY_STOP_KEY = 'early_stopping'
LEARN_TRAINER_KEY = 'trainer'
LEARN_LOSS_KEY = 'loss'
LEARN_OPTIMIZER_KEY = 'optimizer'
LEARN_SCHEDULER_KEY = 'scheduler'

# env vars

AIDENV_NEPT_USER_ENV = 'AIDENV_NEPTUNE_USER'
AIDENV_NEPT_TOKEN_ENV = 'AIDENV_NEPTUNE_TOKEN'
AIDENV_NEPT_PROJECT_ENV = 'AIDENV_NEPTUNE_PROJECT'

# filesystem

TENSORBOARD_DIR = 'tensorboard'
CHECKPOINT_DIR = 'checkpoints'