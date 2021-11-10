import os
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import random_split
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

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

    # custom naming
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

    # log dir creation
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(get_program_output_dir(), 'dlearn')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    name_dir = os.path.join(out_dir, name)
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)
    log_dir = os.path.join(name_dir, dt_string)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    print(f'Log directory is {log_dir}')
    
    log_program(get_program_config(), log_dir)

    return prompt, name, log_dir

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

def build_neptune(config, name):
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
        # load env vars
        user = load_env_var(AIDENV_NEPT_USER_ENV)
        token = load_env_var(AIDENV_NEPT_TOKEN_ENV)
        project = load_env_var(AIDENV_NEPT_PROJECT_ENV)

        # load files to upload
        extensions = search_config_key(config, LOG_NEPT_UP_KEY)
        if not extensions is None:
            source_files = [str(path) for ext in extensions
                            for path in Path('./').rglob(ext)]
        else:
            source_files = None

        # create kwargs
        kwargs['upload_source_files'] = source_files
        kwargs['project_name'] = f'{user}/{project}'
        logger = NeptuneLogger(api_key=token, **kwargs)

    return logger

def build_logging(config, name, log_dir):
    tensorboard = search_config_key(config, LOG_TB_KEY)
    tensorboard = build_tensorboard(tensorboard, name, log_dir)
    print('Loaded tensorboard configuration')

    neptune = search_config_key(config, LOG_NEPT_KEY)
    neptune = build_neptune(neptune, name)
    print('Loaded neptune configuration')

    output = []
    if not tensorboard is None:
        output.append(tensorboard)
    if not neptune is None:
        output.append(neptune)

    return tuple(output)

# core builders

def build_dataset(config, prompt):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG('dataset'))

    return build_dataset_object2(config, prompt)

def build_model(config, prompt):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG('model'))

    model = build_model_object2(config, prompt)
    
    # load from file
    checkpoint = search_config_key(config, CORE_MODEL_CKPT_KEY)
    if not checkpoint is None:
        print('Loading from checkpoint...')
        model.load_from_checkpoint(checkpoint)
        print('Checkpoint loaded')
    
    return model

def build_core(config, prompt):
    dataset = search_config_key(config, CORE_DATASET_KEY)
    dataset = build_dataset(dataset, prompt)
    print('Loaded dataset')

    model = search_config_key(config, CORE_MODEL_KEY)
    model = build_model(model, prompt)
    print('Loaded model')

    return dataset, model

# learning builders

def build_skip(config):
    skip = search_config_key(config, LEARN_SKIP_KEY)

    # default values
    if skip is None:
        skip = { LEARN_TRAIN_KEY : False,
                 LEARN_TEST_KEY : True }
    training = search_config_key(skip, LEARN_TRAIN_KEY)
    if training is None:
        training = False
    test = search_config_key(skip, LEARN_TEST_KEY)
    if test is None:
        test = True

    return training, test

def build_split(config):
    split = search_config_key(config, LEARN_SPLIT_KEY)
    if split is None:
         raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_SPLIT_KEY))

    validation = search_config_key(split, LEARN_VALID_KEY)
    if validation is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_VALID_KEY))

    test = search_config_key(split, LEARN_TEST_KEY)

    return validation, test

def build_data_loaders(config, prompt, dataset, skip, split):
    data_loader = search_config_key(config, LEARN_DATA_LOAD_KEY)
    if data_loader is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG(LEARN_DATA_LOAD_KEY))

    def get_split_indices(d, s):
        d_len = len(d)
        indices = [ int(d_len*(1.-s)),
                    int(d_len*s) ]
        if indices[0] + indices[1] < d_len: # fix float approx
            indices[0] += len - (indices[0]+indices[1])
        return indices

    skip_train, skip_test = skip
    split_valid, split_test = split

    # test
    test_loader = None

    if not skip_test and not split_test is None:
        indices = get_split_indices(dataset, split_test)
        dataset, test_dataset = random_split(dataset, indices)
        test_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[test_dataset])
    else:
        print('Test data loader skipped')

    # training & validation
    train_loader = None
    valid_loader = None

    if not skip_train:
        indices = get_split_indices(dataset, split_valid)
        train_dataset, valid_dataset = random_split(dataset, indices)
        train_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[train_dataset])
        config.shuffle = False
        valid_loader =  build_learning_object1(config, prompt,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[valid_dataset])
    else:
        print('Train and validation data loader skipped')

    return train_loader, valid_loader, test_loader

def build_early_stopping(config, prompt):
    early_stopping = search_config_key(config, LEARN_EARLY_STOP_KEY)
    if early_stopping is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_EARLY_STOP_KEY))

    return build_learning_object1(config, prompt,
                                    LEARN_EARLY_STOP_KEY)

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

def build_loss(config, prompt):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_LOSS_KEY))

    return build_learning_object2(config, prompt)

def build_optimizer(config, prompt, model):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_OPTIMIZER_KEY))

    return build_learning_object2(config, prompt,
                                    args=[model.parameters()])

def build_scheduler(config, prompt, optimizer):
    if config is None:
        return None

    return build_learning_object2(config, prompt,
                                    args=[optimizer])
    
def build_learning(config, prompt, dataset, model, loggers, log_dir):
    skip = build_skip(config)
    print('Loaded skip options')

    split = build_split(config)
    print('Loaded split options')

    loaders = build_data_loaders(config, prompt, dataset, skip, split)
    print('Loaded data loaders')
    yield loaders

    early_stop = build_early_stopping(config, prompt)
    print('Loaded early stopping')

    ckpt_path = os.path.join(log_dir, CHECKPOINT_DIR)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [ early_stop, checkpoint_callback, lr_monitor ]

    trainer = build_trainer(config, prompt, callbacks, loggers, log_dir)
    print('Loaded trainer')
    yield trainer

    loss = search_config_key(config, LEARN_LOSS_KEY)
    loss = build_loss(loss, prompt)
    print('Loaded loss')
    yield loss

    optimizer = search_config_key(config, LEARN_OPTIMIZER_KEY)
    optimizer = build_optimizer(optimizer, prompt, model)
    print('Loaded optimizer')
    yield optimizer

    scheduler = search_config_key(config, LEARN_SCHEDULER_KEY)
    scheduler = build_scheduler(scheduler, prompt, optimizer)
    print('Loaded scheduler')
    yield scheduler