import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.string import lower_identifier
from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.params import CLASS_NAME_KEY
from aidenv.app.params import DATASET_MODULE
from aidenv.app.params import MODEL_MODULE
from aidenv.app.params import LEARNING_MODULE
from aidenv.app.params import set_core_package
from aidenv.app.config import get_program_output_dir
from aidenv.app.config import get_program_config
from aidenv.app.config import search_config_key
from aidenv.app.config import load_env_var
from aidenv.app.config import build_simple_object1
from aidenv.app.config import build_simple_object2
from aidenv.app.config import build_core_object1
from aidenv.app.config import build_core_object2
from aidenv.app.logging import log_program
from aidenv.app.dlearn.params import *
from aidenv.app.dlearn.logging import LoggerCollection

# executable objects builders

def build_dataset_object1(config, prompt):
    '''
    Build an expanded dataset object.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object1(config, prompt, DATASET_MODULE)

def build_dataset_object2(config, prompt, kwargs={}):
    '''
    Build an expanded dataset object.
    This object exploits core prompt for locating the class name.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object2(config, prompt, DATASET_MODULE, True,
                                kwargs)

def build_model_object1(config, prompt):
    '''
    Build an expanded model object.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object1(config, prompt, MODEL_MODULE)

def build_model_object2(config, prompt, kwargs={}):
    '''
    Build an expanded model object.
    This object exploits core prompt for locating the class name.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object2(config, prompt, MODEL_MODULE, False,
                                kwargs)

def build_learning_object1(config, prompt, class_name, args=[], kwargs={}):
    '''
    Build a collapsed learning object.
    config: configuration object
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    class_name: name of the class
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    '''
    return build_simple_object1(config, prompt, LEARNING_MODULE,
                                class_name, args, kwargs)

def build_learning_object2(config, prompt, args=[], kwargs={}):
    '''
    Build an expanded learning object.
    config: configuration object
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    '''
    return build_simple_object2(config, prompt, LEARNING_MODULE,
                                    args, kwargs)

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

    env_dir = os.path.join(out_dir, 'dlearn')
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
    force = search_config_key(config, DETERM_FORCE_KEY)
    if force is None:
        force = False
        print('Not forcing determinism')

    seed = search_config_key(config, DETERM_SEED_KEY)
    if seed is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_SEED_KEY))

    if force:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        seed_everything(seed, workers=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        workers = search_config_key(config, DETERM_WORKERS_KEY)
        if workers is None:
            workers = True
            print('Seeding workers')

        seed_everything(seed, workers=workers)

# core builders

def build_skip(config, hparams):
    skip = search_config_key(config, CORE_DATASET_TRAIN_KEY)
    skip_params = {}

    if skip is None:
        skip_params[CORE_DATASET_TRAIN_KEY] = False
        skip_params[CORE_DATASET_TEST_KEY] = False
        hparams.update({CORE_DATASET_SKIP_KEY : skip_params})
        return False, False

    # train
    train = search_config_key(skip, CORE_DATASET_TRAIN_KEY)
    if train is None:
        skip_params[CORE_DATASET_TRAIN_KEY] = False
    else:
        skip_params[CORE_DATASET_TRAIN_KEY] = train

    # test
    test = search_config_key(skip, CORE_DATASET_TEST_KEY)
    if test is None:
        skip_params[CORE_DATASET_TRAIN_KEY] = False
    else:
        skip_params[CORE_DATASET_TEST_KEY] = test

    # hyperparams
    hparams.update({CORE_DATASET_SKIP_KEY : skip_params})

    return train, test

INVALID_SKIPS_MSG = 'both training and test cannot be skipped'
INVALID_TRAIN_VALID_MSG = 'sum of sizes of train/valid is greater than chunks count'
INVALID_TEST_MSG = 'size of test is 0'
INVALID_SUBSET_MSG = 'dataset subset size type can only be int or float'

def build_dataset(config, prompt, hparams):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG('dataset'))

    skip_train, skip_test = build_skip(config, hparams)
    if skip_train and skip_test:
        raise ValueError(INVALID_SKIPS_MSG)

    train_set = None
    valid_set = None
    test_set = None

    if not skip_train:
        train_set = build_dataset_object2(config, prompt,
                                            kwargs={'set_name':'train'})
        valid_set = build_dataset_object2(config, prompt,
                                            kwargs={'set_name':'valid'})
    else:
        print('Training will be skipped')

    if not skip_test:
        test_set = build_dataset_object2(config, prompt,
                                            kwargs={'set_name':'test'})
    else:
        print('Testing will be skipped')

    print(f'Training set has size {len(train_set)}')
    print(f'Validation set has size {len(valid_set)}')
    print(f'Test set has size {len(test_set)}')
    
    hparams.update(dict(config))
    datasets = (train_set, valid_set, test_set)
    return datasets

def build_model(config, prompt, hparams):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG('model'))

    model = build_model_object2(config, prompt)
    
    checkpoint = search_config_key(config, CORE_MODEL_CKPT_KEY)
    if not checkpoint is None:
        print('Loading from checkpoint...')
        model.load_from_checkpoint(checkpoint)
        print('Checkpoint loaded')

    hparams.update(dict(config))
    return model

def build_core(config, hparams, prompt):
    core_hparams = {}

    dataset = search_config_key(config, CORE_DATASET_KEY)
    dataset_hparams = {}
    datasets = build_dataset(dataset, prompt, dataset_hparams)
    core_hparams[CORE_DATASET_KEY] = dataset_hparams
    print('Loaded dataset')

    model = search_config_key(config, CORE_MODEL_KEY)
    model_hparams = {}
    model = build_model(model, prompt, model_hparams)
    core_hparams[CORE_MODEL_KEY] = model_hparams
    print('Loaded model')

    hparams.update(core_hparams)

    return datasets, model

# learning1 builders

def build_loss(config, prompt, hparams):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_LOSS_KEY))

    hparams.update(dict(config))
    return build_learning_object2(config, prompt)

def build_early_stopping(config, prompt, hparams):
    early_stopping = search_config_key(config, LEARN_EARLY_STOP_KEY)
    if early_stopping is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_EARLY_STOP_KEY))

    hparams.update({ LEARN_EARLY_STOP_KEY: dict(early_stopping) })
    return build_learning_object1(config, prompt,
                                    LEARN_EARLY_STOP_KEY)

def build_optimizer(config, prompt, hparams, model):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_OPTIMIZER_KEY))

    hparams.update(dict(config))
    return build_learning_object2(config, prompt,
                                    args=[model.parameters()])

def build_scheduler(config, prompt, hparams, optimizer):
    if config is None:
        return None

    hparams.update(dict(config))
    return build_learning_object2(config, prompt,
                                    args=[optimizer])

def build_data_loaders(config, prompt, hparams, datasets):
    data_loader = search_config_key(config, LEARN_DATA_LOAD_KEY)
    if data_loader is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG(LEARN_DATA_LOAD_KEY))

    shuffle = data_loader.shuffle
    train_set, valid_set, test_set = datasets

    # training
    train_loader = None
    if not train_set is None:
        train_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[train_set])

    # validation
    valid_loader = None
    if not valid_set is None:
        data_loader.shuffle = False
        valid_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[valid_set])

    # test
    test_loader = None
    if not test_set is None:
        data_loader.shuffle = False
        test_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[test_set])

    data_loader.shuffle = shuffle
    hparams.update({ LEARN_DATA_LOAD_KEY : dict(data_loader) })
    return train_loader, valid_loader, test_loader
    
def build_learning1(config, hparams, prompt, datasets, model):
    learning_hparams = {}

    loss = search_config_key(config, LEARN_LOSS_KEY)
    loss_hparams = {}
    loss = build_loss(loss, prompt, loss_hparams)
    learning_hparams[LEARN_LOSS_KEY] = loss_hparams
    print('Loaded loss')

    early_stop_hparams = {}
    early_stop = build_early_stopping(config, prompt, early_stop_hparams)
    learning_hparams.update(early_stop_hparams)
    print('Loaded early stopping')

    optimizer = search_config_key(config, LEARN_OPTIMIZER_KEY)
    optimizer_hparams = {}
    optimizer = build_optimizer(optimizer, prompt, optimizer_hparams, model)
    learning_hparams[LEARN_OPTIMIZER_KEY] = optimizer_hparams
    print('Loaded optimizer')

    scheduler = search_config_key(config, LEARN_SCHEDULER_KEY)
    scheduler_hparams = {}
    scheduler = build_scheduler(scheduler, prompt, scheduler_hparams, optimizer)
    learning_hparams[LEARN_SCHEDULER_KEY] = scheduler_hparams
    print('Loaded scheduler')

    loaders_hparams = {}
    loaders = build_data_loaders(config, prompt, loaders_hparams, datasets)
    learning_hparams.update(loaders_hparams)
    print('Loaded data loaders')

    hparams.update(learning_hparams)

    return early_stop, loss, optimizer, scheduler, loaders

# logging builders

def build_tensorboard(config, name, log_dir):
    if config is None:
        return None
    
    kwargs = {}
    kwargs.update(dict(config))
    del kwargs[LOG_ENABLE_KEY]
    logger = None

    enable = search_config_key(config, LOG_ENABLE_KEY)
    if enable is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LOG_ENABLE_KEY))

    if config.enable:
        kwargs['name'] = name
        kwargs['save_dir'] = os.path.join(log_dir, TENSORBOARD_DIR)
        logger = TensorBoardLogger(**kwargs)

    return logger

def build_neptune(config, name, id, descr, tags):
    if config is None:
        return None

    kwargs = {}
    kwargs.update(dict(config))
    del kwargs[LOG_ENABLE_KEY]
    logger = None

    enable = search_config_key(config, LOG_ENABLE_KEY)
    if enable is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LOG_ENABLE_KEY))

    if enable:
        user = load_env_var(AIDENV_NEPT_USER_ENV)
        token = load_env_var(AIDENV_NEPT_TOKEN_ENV)
        project = load_env_var(AIDENV_NEPT_PROJECT_ENV)

        kwargs['prefix'] = 'experiment'
        kwargs['project'] = f'{user}/{project}'
        kwargs['name'] = f'{name}_{id}'
        kwargs['custom_run_id'] = id
        kwargs['description'] = descr
        kwargs['tags'] = tags + list(kwargs['tags'])

        logger = NeptuneLogger(api_key=token, **kwargs)

    return logger

def build_logging(config, hparams, origin, prompt, name, id, log_dir, descr):
    # tensorboard
    tensorboard = search_config_key(config, LOG_TB_KEY)
    tensorboard = build_tensorboard(tensorboard, name, log_dir)
    if not tensorboard is None:
        # tensorboard.log_hyperparams(hparams)      # saving whole config file
        print('Loaded tensorboard')

    # neptune
    tag = origin
    for p in prompt:
        tag = f'{tag}-{p}'
    tags = [tag]            # only tag is cat origin+prompt

    neptune = search_config_key(config, LOG_NEPT_KEY)
    neptune = build_neptune(neptune, name, id, descr, tags)
    if not neptune is None:
        neptune.log_hyperparams(hparams)
        print('Loaded neptune')

    output = []
    if not tensorboard is None:
        output.append(tensorboard)
    if not neptune is None:
        output.append(neptune)

    if len(output) == 0:
        raise ValueError('You must specify at least one logger')

    return tuple(output)

# learning2 builder

def build_trainer(config, prompt, callbacks, loggers, log_dir):
    trainer = search_config_key(config, LEARN_TRAINER_KEY)
    if trainer is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_TRAINER_KEY))

    loggers = LoggerCollection(loggers)

    return build_learning_object1(config, prompt,
                                    LEARN_TRAINER_KEY,
                                    kwargs= {
                                        'logger' : loggers,
                                        'callbacks' : callbacks,
                                        'default_root_dir' : log_dir
                                    })

def build_loggables(config, prompt):
    if config is None:
        return []

    loggables = {}
    sets = {}
    for c in config:
        cls_name = c[CLASS_NAME_KEY]
        c[CLASS_NAME_KEY] = f'Loggable{cls_name}'

        log_name = lower_identifier(cls_name)
        loggables[log_name] = build_learning_object2(c, prompt)
        ss = search_config_key(c, LEARN_SETS_KEY)
        if ss is None:
            ss = ['train','valid','test']
        else:
            ss = list(ss)
        sets[log_name] = ss

    return loggables, sets

def build_learning2(config, prompt, early_stop, loggers, log_dir):
    ckpt_path = os.path.join(log_dir, CHECKPOINT_DIR)
    ckpt_callback = ModelCheckpoint(dirpath=ckpt_path, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [ early_stop, ckpt_callback, lr_monitor ]
    trainer = build_trainer(config, prompt, callbacks, loggers, log_dir)
    print('Loaded trainer')

    loggables = search_config_key(config, LEARN_LOG_KEY)
    loggables, sets = build_loggables(loggables, prompt)
    print('Loaded loggables')

    return trainer, loggables, sets