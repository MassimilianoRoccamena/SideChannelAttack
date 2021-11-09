import os
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import random_split
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from aidenv.app.params import LOG_DIR
from aidenv.app.params import DATASET_MODULE
from aidenv.app.params import MODEL_MODULE
from aidenv.app.params import LEARNING_MODULE
from aidenv.app.params import set_core_package
from aidenv.app.config import load_config
from aidenv.app.config import build_simple_object1
from aidenv.app.config import build_simple_object2
from aidenv.app.config import build_core_object1
from aidenv.app.config import build_core_object2
from aidenv.app.dlearn.params import EXECUTABLE_CONF_PATH
from aidenv.app.dlearn.params import TENSORBOARD_DIR
from aidenv.app.dlearn.params import CHECKPOINT_DIR
from aidenv.app.dlearn.params import NEPTUNE_PROJECT_ENV
from aidenv.app.dlearn.params import NEPTUNE_USER_ENV
from aidenv.app.dlearn.params import NEPTUNE_TOKEN_ENV
from aidenv.app.dlearn.logging import LoggerCollection
from aidenv.app.dlearn.logging import HyperParamsLogger

# loading and definitions

def load_training_config():
    return load_config(EXECUTABLE_CONF_PATH)

CONFIG_NOT_FOUND_MSG = lambda id: f'{id} configuration not found'

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
    # load core origin
    origin = config.origin
    if origin is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('origin'))
    set_core_package(list(origin))

    # load core prompt
    prompt = config.prompt
    if prompt is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('prompt'))

    # custom naming
    name = config.name
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
    log_dir = os.path.join(LOG_DIR, name, dt_string)
    print(f'Log directory is {log_dir}')

    return prompt, name, log_dir

# determinism builders

def build_determinism(config):
    if config.force_determinism is None:
        config.force_determinism = False
        print('setting force determinism to false')

    if config.force_determinism:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        seed_everything(config.seed, workers=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    elif not config.seed is None:
        seed_everything(config.seed, workers=config.seed_workers)

# logging builders

def build_tensorboard(config, name, log_dir):
    if config is None:
        return None
    
    kwargs = {}
    kwargs.update(dict(config))
    del kwargs['enable']
    logger = None

    if config.enable is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('enable'))

    if config.enable:
        kwargs['name'] = name
        kwargs['save_dir'] = os.path.join(log_dir, TENSORBOARD_DIR)
        logger = TensorBoardLogger(**kwargs)

    return logger

ENV_NOT_FOUND_MSG = lambda env: f'{env} environment variable not found'

def build_neptune(config, name):
    if config is None:
        return None

    kwargs = {}
    kwargs.update(dict(config))
    del kwargs['enable']
    logger = None

    if config.enable is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('enable'))

    if config.enable:
        # load env vars
        try:
            user = os.environ[NEPTUNE_USER_ENV]
        except KeyError:
            raise RuntimeError(ENV_NOT_FOUND_MSG(NEPTUNE_USER_ENV))
        try:
            token = os.environ[NEPTUNE_TOKEN_ENV]
        except KeyError:
            raise RuntimeError(ENV_NOT_FOUND_MSG(NEPTUNE_TOKEN_ENV))
        try:
            project = os.environ[NEPTUNE_PROJECT_ENV]
        except KeyError:
            raise RuntimeError(ENV_NOT_FOUND_MSG(NEPTUNE_PROJECT_ENV))

        # load files to upload
        extensions = config.upload_source_files
        if extensions:
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
    tensorboard = build_tensorboard(config.tensorboard, name, log_dir)
    print('Loaded tensorboard configuration')

    neptune = build_neptune(config.neptune, name)
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
        raise ValueError(CONFIG_NOT_FOUND_MSG('dataset'))

    return build_dataset_object2(config, prompt)

def build_model(config, prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('model'))

    model = build_model_object2(config, prompt)
    
    # load from file
    checkpoint = config.checkpoint
    if not checkpoint is None:
        print('Loading from checkpoint...')
        model.load_from_checkpoint(checkpoint)
        print('Checkpoint loaded')
    
    return model

def build_core(config, prompt):
    dataset = build_dataset(config.dataset, prompt)
    print('Loaded dataset')

    model = build_model(config.model, prompt)
    print('Loaded model')

    return dataset, model

# learning builders

def build_skip(config):
    skip = config.skip

    if skip is None:
        skip = {'training':False, 'test':True}
    if skip.training is None:
        skip.training = False
    if skip.test is None:
        skip.test = True

    return skip.training, skip.test

def build_split(config):
    split = config.split

    if split is None:
         raise ValueError(CONFIG_NOT_FOUND_MSG('split'))
    if split.validation is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('validation split'))

    return split.validation, split.test

def build_data_loaders(config, prompt, dataset, skip, split):
    if config.data_loader is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('data loader'))

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
                                                'data_loader',
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
                                                'data_loader',
                                                args=[train_dataset])
        config.shuffle = False
        valid_loader =  build_learning_object1(config, prompt,
                                                'data_loader',
                                                args=[valid_dataset])
    else:
        print('Train and validation data loader skipped')

    return train_loader, valid_loader, test_loader

def build_early_stopping(config, prompt):
    if config.early_stopping is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('early stopping'))

    return build_learning_object1(config, prompt,
                                    'early_stopping')

def build_trainer(config, prompt, callbacks, loggers, log_dir):
    if config.trainer is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('trainer'))

    loggers = LoggerCollection(loggers)

    return build_learning_object1(config, prompt,
                                    'trainer',
                                    kwargs= {
                                        'logger' : loggers,
                                        'callbacks' : callbacks,
                                        'default_root_dir' : log_dir
                                    })

def build_loss(config, prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('loss'))

    return build_learning_object2(config, prompt)

def build_optimizer(config, prompt, model):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('optimizer'))

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

    params_writer = HyperParamsLogger(config, log_dir, 'params.yaml')
    ckpt_path = os.path.join(log_dir, CHECKPOINT_DIR)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [ early_stop, params_writer, checkpoint_callback, lr_monitor ]

    trainer = build_trainer(config, prompt, callbacks, loggers, log_dir)
    print('Loaded trainer')
    yield trainer

    loss = build_loss(config.loss, prompt)
    print('Loaded loss')
    yield loss

    optimizer = build_optimizer(config.optimizer, prompt, model)
    print('Loaded optimizer')
    yield optimizer

    scheduler = build_scheduler(config.scheduler, prompt, optimizer)
    print('Loaded scheduler')
    yield scheduler