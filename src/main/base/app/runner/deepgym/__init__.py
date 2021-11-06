import os
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import random_split
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from main.base.app.params import LOG_DIR
from main.base.app.params import DATASET_MODULE
from main.base.app.params import MODEL_MODULE
from main.base.app.params import LEARNING_MODULE
from main.base.app.config import load_config
from main.base.app.config import build_simple_object1, build_simple_object2
from main.base.app.config import build_core_object2
from main.base.app.runner.deepgym.params import TENSORBOARD_DIR
from main.base.app.runner.deepgym.params import EXECUTABLE_CONF_PATH
from main.base.app.runner.deepgym.params import NEPTUNE_PRJ_NAME
from main.base.app.runner.deepgym.params import NEPTUNE_USER_ENV
from main.base.app.runner.deepgym.params import NEPTUNE_TOKEN_ENV
from main.base.app.runner.deepgym.logging import LoggerCollection, HyperParamsLogger

# loading and definitions

def load_training_config():
    return load_config(EXECUTABLE_CONF_PATH)

CONFIG_NOT_FOUND_MSG = lambda id: f'{id} configuration not found'

# base builders

INVALID_PROMPT_MSG = 'selected prompt not valid'

def build_base(config):
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
        print('using default experiment name')

    # log dir creation
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOG_DIR, name, dt_string)
    print(f'log directory is {log_dir}')

    # optional skip
    skip = config.skip
    if skip is None:
        skip = {'training':False, 'testing':True}
    if skip.training is None:
        skip.training = False
    if skip.testing is None:
        skip.testing = True

    return prompt, name, log_dir, skip

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
        # check env vars
        try:
            user = os.environ[NEPTUNE_USER_ENV]
        except KeyError:
            raise RuntimeError(ENV_NOT_FOUND_MSG(NEPTUNE_USER_ENV))
        try:
            token = os.environ[NEPTUNE_TOKEN_ENV]
        except KeyError:
            raise RuntimeError(ENV_NOT_FOUND_MSG(NEPTUNE_TOKEN_ENV))

        # load files to upload
        extensions = config.upload_source_files
        if extensions:
            source_files = [str(path) for ext in extensions
                            for path in Path('./').rglob(ext)]
        else:
            source_files = None

        # create kwargs
        kwargs['upload_source_files'] = source_files
        kwargs['experiment_name'] = name
        if config.project_name is None:
            project_name = NEPTUNE_PRJ_NAME
            print('using default neptune project name')
        else:
            project_name = config.project_name
        kwargs['project_name'] = f'{user}/{project_name}'
        logger = NeptuneLogger(api_key=token, **kwargs)

    return logger

def build_logging(config, name, log_dir):
    tensorboard = build_tensorboard(config.tensorboard, name, log_dir)
    neptune = build_neptune(config.neptune, name)

    output = []
    if not tensorboard is None:
        output.append(tensorboard)
    if not neptune is None:
        output.append(neptune)

    return tuple(output)

# core builder

def build_dataset(config, prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('dataset'))

    return build_core_object2(config, prompt, DATASET_MODULE)

def build_model(config, prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('model'))

    model = build_core_object2(config, prompt,
                                MODEL_MODULE, prompt_suffix=False)
    
    # load from file
    checkpoint = config.checkpoint
    if not checkpoint is None:
        print('loading from checkpoint...')
        model.load_from_checkpoint(checkpoint)
        print('checkpoint loaded')
    
    return model

def build_core(config, prompt):
    dataset = build_dataset(config.dataset, prompt)
    assert not dataset is None
    print('loaded dataset')

    model = build_model(config.model, prompt)
    assert not model is None
    print('loaded model')

    return dataset, model

# learning builders

def build_data_loaders(config, prompt, dataset):
    valid_split = config.validation_split
    if valid_split is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('validation split'))

    if config.data_loader is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('data loader'))

    # split indices
    data_len = len(dataset)
    indices = [ int(data_len*(1.-valid_split)),
                int(data_len*valid_split) ]
    if indices[0] + indices[1] < data_len: # fix float approx
        indices[0] += data_len - (indices[0]+indices[1])

    # do split
    train_dataset, valid_dataset = random_split(dataset, indices)
    train_loader =  build_simple_object1(config, prompt,
                                            LEARNING_MODULE, 'data_loader',
                                            args=[train_dataset])
    config.shuffle = False
    valid_loader =  build_simple_object1(config, prompt,
                                            LEARNING_MODULE, 'data_loader',
                                            args=[valid_dataset])   

    return train_loader, valid_loader

def build_early_stopping(config, prompt):
    if config.early_stopping is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('early stopping'))

    return build_simple_object1(config, prompt,
                                    LEARNING_MODULE, 'early_stopping')

def build_trainer(config, prompt, loggers, callbacks):
    if config.trainer is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('trainer'))

    loggers = LoggerCollection(loggers)

    return build_simple_object1(config, prompt,
                                LEARNING_MODULE, 'trainer',
                                kwargs= {
                                    'logger' : loggers,
                                    'callbacks' : callbacks
                                })

def build_loss(config, prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('loss'))

    return build_simple_object2(config, prompt, LEARNING_MODULE)

def build_optimizer(config, prompt, model):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('optimizer'))

    return build_simple_object2(config, prompt, LEARNING_MODULE,
                                    args=[model.parameters()])

def build_scheduler(config, prompt, optimizer):
    if config is None:
        return None

    return build_simple_object2(config, prompt, LEARNING_MODULE,
                                    args=[optimizer])
    
def build_learning(config, prompt, dataset, model, loggers):
    train_loader, valid_loader = build_data_loaders(config, prompt, dataset)
    assert not train_loader is None
    assert not valid_loader is None
    print('loaded data loaders')
    yield (train_loader, valid_loader)

    early_stop = build_early_stopping(config, prompt)
    assert not early_stop is None
    print('loaded early stopping')

    params_writer = HyperParamsLogger(config, './', 'training.yaml')
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [ early_stop, params_writer, checkpoint_callback, lr_monitor ]

    trainer = build_trainer(config, prompt, loggers, callbacks)
    assert not trainer is None
    print('loaded trainer')
    yield trainer

    loss = build_loss(config.loss, prompt)
    assert not loss is None
    print('loaded loss')
    yield loss

    optimizer = build_optimizer(config.optimizer, prompt, model)
    assert not optimizer is None
    print('loaded optimizer')
    yield optimizer

    scheduler = build_scheduler(config.scheduler, prompt, optimizer)
    assert not scheduler is None
    print('loaded scheduler')
    yield scheduler

# section parsing

def parse_base(config):
    config = config.base
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('base'))

    prompt, name, log_dir, skip = build_base(config)
    
    print('base configuration done')
    return prompt, name, log_dir, skip

def parse_determinism(config):
    config = config.determinism
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('determinism'))

    build_determinism(config)

    print('determinism configuration done')

def parse_logging(config, name, log_dir):
    config = config.logging
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('logging'))

    loggers = build_logging(config, name, log_dir)

    print("logging configuration done")
    return loggers

def parse_core(config, prompt):
    config = config.core
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('core'))

    dataset, model = build_core(config, prompt)

    print("core configuration done")
    return dataset, model

def parse_learning(config, prompt, dataset, model, loggers):
    config = config.learning
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('learning'))

    learning = build_learning(config, prompt,
                                dataset, model, loggers)

    train_loader, valid_loader = next(learning)
    trainer = next(learning)
    loss = next(learning)
    optimizer = next(learning)
    scheduler = next(learning)

    model.set_learning(loss, optimizer, scheduler=scheduler)
    model.mount_from_dataset(dataset)

    print('learning configuration done')
    return train_loader, valid_loader, trainer

# main

def run_train_test(trainer, model, train_loader, valid_loader, test_loader):
    if not (train_loader is None or valid_loader is None):
        trainer.fit(model, train_loader, valid_loader)
        print('model training done')
    else:
        print('model training skipped')

    if not test_loader is None:
        trainer.test(model, test_loader)
        print('model testing done')
    else:
        print('model testing skipped')

def run():
    '''
    Entry point for deep-gym executable
    '''
    print("deep gym started")

    config = load_training_config()
    prompt, name, log_dir, skip = parse_base(config)

    if skip['training'] and skip['testing']:
        print('found nothing to do')
    else:
        parse_determinism(config)
        loggers = parse_logging(config, name, log_dir)

        dataset, model = parse_core(config, prompt)
        train, valid, trainer = parse_learning(config, prompt, dataset,
                                                model, loggers)

        run_train_test(trainer, model, train, valid, None) # test WIP
    
    print('deep gym finished')