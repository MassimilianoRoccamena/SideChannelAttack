import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data import random_split
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from aidenv.app.params import CONFIG_NOT_FOUND_MSG
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

def build_dataset_object2(config, prompt):
    '''
    Build an expanded dataset object.
    This object exploits core prompt for locating the class name.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object2(config, prompt, DATASET_MODULE, True)

def build_model_object1(config, prompt):
    '''
    Build an expanded model object.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object1(config, prompt, MODEL_MODULE)

def build_model_object2(config, prompt):
    '''
    Build an expanded model object.
    This object exploits core prompt for locating the class name.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_core_object2(config, prompt, MODEL_MODULE, False)

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
            name = f"{p[0]}{p[2]}{p[1]}"
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

    print(f'Log directory is {log_dir}')
    log_program(get_program_config(), log_dir)
    print('Stored program configuration')

    # description
    descr = search_config_key(config, BASE_DESCR_KEY)
    if prompt is None:
        descr = ''

    return prompt, name, id, log_dir, descr

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

def build_dataset(config, prompt, hparams):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG('dataset'))

    dataset = build_dataset_object2(config, prompt)

    # sampling
    nsamples = search_config_key(config, CORE_DATASET_NSAMP_KEY)

    # hyperparams
    hparams.update(dict(config))

    return dataset, nsamples

def build_model(config, prompt, hparams):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG('model'))

    model = build_model_object2(config, prompt)
    
    # checkpoint
    checkpoint = search_config_key(config, CORE_MODEL_CKPT_KEY)
    if not checkpoint is None:
        print('Loading from checkpoint...')
        model.load_from_checkpoint(checkpoint)
        print('Checkpoint loaded')

    # hyperparams
    hparams.update(dict(config))
    
    return model

def build_core(config, hparams, prompt):
    core_hparams = {}

    dataset = search_config_key(config, CORE_DATASET_KEY)
    dataset_hparams = {}
    dataset, nsamples = build_dataset(dataset, prompt, dataset_hparams)
    core_hparams[CORE_DATASET_KEY] = dataset_hparams
    print('Loaded dataset')

    model = search_config_key(config, CORE_MODEL_KEY)
    model_hparams = {}
    model = build_model(model, prompt, model_hparams)
    core_hparams[CORE_MODEL_KEY] = model_hparams
    print('Loaded model')

    hparams.update(core_hparams)

    return dataset, nsamples, model

# learning1 builders

def build_skip(config, hparams):
    skip = search_config_key(config, LEARN_SKIP_KEY)

    # default
    if skip is None:
        skip = { LEARN_TRAIN_KEY : False,
                 LEARN_TEST_KEY : True }
    training = search_config_key(skip, LEARN_TRAIN_KEY)
    if training is None:
        training = False
    test = search_config_key(skip, LEARN_TEST_KEY)
    if test is None:
        test = True

    # hyperparams
    hparams.update({ LEARN_SKIP_KEY : 
                     { LEARN_TRAIN_KEY : training,
                       LEARN_TEST_KEY : test }})

    return training, test

def build_split(config, hparams):
    split = search_config_key(config, LEARN_SPLIT_KEY)

    # default
    if split is None:
         raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_SPLIT_KEY))

    validation = search_config_key(split, LEARN_VALID_KEY)
    if validation is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_VALID_KEY))

    test = search_config_key(split, LEARN_TEST_KEY)

    # hyperparams
    hparams.update({ LEARN_SPLIT_KEY : 
                     { LEARN_VALID_KEY : validation,
                       LEARN_TEST_KEY : test }})

    return validation, test

def build_early_stopping(config, prompt, hparams):
    early_stopping = search_config_key(config, LEARN_EARLY_STOP_KEY)

    # default
    if early_stopping is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_EARLY_STOP_KEY))

    # hyperparams
    hparams.update({ LEARN_EARLY_STOP_KEY: dict(early_stopping) })

    return build_learning_object1(config, prompt,
                                    LEARN_EARLY_STOP_KEY)

def build_loss(config, prompt, hparams):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_LOSS_KEY))

    # hyperparams
    hparams.update(dict(config))

    return build_learning_object2(config, prompt)

def build_optimizer(config, prompt, hparams, model):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_OPTIMIZER_KEY))

    # hyperparams
    hparams.update(dict(config))

    return build_learning_object2(config, prompt,
                                    args=[model.parameters()])

def build_scheduler(config, prompt, hparams, optimizer):
    if config is None:
        return None

    # hyperparams
    hparams.update(dict(config))

    return build_learning_object2(config, prompt,
                                    args=[optimizer])

def build_data_loaders(config, prompt, hparams, dataset, nsamples, skip, split):
    data_loader = search_config_key(config, LEARN_DATA_LOAD_KEY)
    if data_loader is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG(LEARN_DATA_LOAD_KEY))

    shuffle = data_loader.shuffle

    # split lengths
    def get_split_lengths(d, s):
        d_len = len(d)
        len1 = int(d_len * (1.-s))
        len2 = d_len - len1
        return [len1, len2]

    skip_train, skip_test = skip
    split_valid, split_test = split

    # sampling
    if not nsamples is None:
        print('Sampling the dataset')
        indices = np.random.choice(len(dataset), nsamples, replace=False)
        dataset = Subset(dataset, indices)

    # test
    test_loader = None

    if not skip_test and not split_test is None:
        length = get_split_lengths(dataset, split_test)
        dataset, test_dataset = random_split(dataset, length)
        print(f'Optimization set has size {len(dataset)}')
        print(f'Test set has size {len(test_dataset)}')

        data_loader.shuffle = False
        test_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[test_dataset])
    else:
        print('Test data loader skipped')

    # training & validation
    train_loader = None
    valid_loader = None

    if not skip_train:
        length = get_split_lengths(dataset, split_valid)
        train_dataset, valid_dataset = random_split(dataset, length)
        print(f'Training set has size {len(train_dataset)}')
        print(f'Validation set has size {len(valid_dataset)}')

        data_loader.shuffle = shuffle
        train_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[train_dataset])
        data_loader.shuffle = False
        valid_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[valid_dataset])

        data_loader.shuffle = shuffle
    else:
        print('Train and validation data loader skipped')

    # hyperparams
    hparams.update({ LEARN_DATA_LOAD_KEY : dict(data_loader) })

    return train_loader, valid_loader, test_loader
    
def build_learning1(config, hparams, prompt, dataset, nsamples, model):
    learning_hparams = {}

    skip_hparams = {}
    skip = build_skip(config, skip_hparams)
    learning_hparams.update(skip_hparams)
    print('Loaded skip options')

    split_hparams = {}
    split = build_split(config, split_hparams)
    learning_hparams.update(split_hparams)
    print('Loaded split options')

    early_stop_hparams = {}
    early_stop = build_early_stopping(config, prompt, early_stop_hparams)
    learning_hparams.update(early_stop_hparams)
    print('Loaded early stopping')

    loss = search_config_key(config, LEARN_LOSS_KEY)
    loss_hparams = {}
    loss = build_loss(loss, prompt, loss_hparams)
    learning_hparams[LEARN_LOSS_KEY] = loss_hparams
    print('Loaded loss')

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
    loaders = build_data_loaders(config, prompt, loaders_hparams,
                                    dataset, nsamples, skip, split)
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
        # load env vars
        user = load_env_var(AIDENV_NEPT_USER_ENV)
        token = load_env_var(AIDENV_NEPT_TOKEN_ENV)
        project = load_env_var(AIDENV_NEPT_PROJECT_ENV)

        # neptune kwargs
        kwargs['prefix'] = 'learning'
        kwargs['project'] = f'{user}/{project}'
        kwargs['name'] = name
        kwargs['custom_run_id'] = id
        kwargs['description'] = descr
        kwargs['tags'] = tags

        logger = NeptuneLogger(api_key=token, **kwargs)

    return logger

def build_logging(config, hparams, prompt, name, id, log_dir, descr):
    # tensorboard
    tensorboard = search_config_key(config, LOG_TB_KEY)
    tensorboard = build_tensorboard(tensorboard, name, log_dir)
    if not tensorboard is None:
        # tensorboard.log_hyperparams(hparams)      # saving whole config file
        print('Loaded tensorboard')

    # neptune
    tag = prompt[0]
    for p in prompt[1:]:
        tag = f'{tag}-{p}'
    tags = [tag]            # only tag is cat prompt

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
        print('No logger has been found')

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

def build_learning2(config, prompt, early_stop, loggers, log_dir):
    ckpt_path = os.path.join(log_dir, CHECKPOINT_DIR)
    ckpt_callback = ModelCheckpoint(dirpath=ckpt_path, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [ early_stop, ckpt_callback, lr_monitor ]

    trainer = build_trainer(config, prompt, callbacks, loggers, log_dir)
    print('Loaded trainer')
    return trainer