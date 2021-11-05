import os
import torch
from torch.utils.data import random_split
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from main.base.app.params import CONFIG_DIR
from main.base.app.params import LOGS_DIR
from main.base.app.params import TENSORBOARD_DIR
from main.base.app.params import NEPTUNE_DIR
from main.base.app.params import DATASET_MODULE
from main.base.app.params import MODEL_MODULE
from main.base.app.params import LEARNING_MODULE
from main.base.app.config import load_config, build_core_prompt
from main.base.app.config import build_simple_object1, build_simple_object2
from main.base.app.config import build_core_object2

# basic

TRAINING_CONF_PATH = os.path.join(CONFIG_DIR, 'training.yaml')

def load_training_config():
    return load_config(TRAINING_CONF_PATH)

CONFIG_NOT_FOUND_MSG = lambda id: f'{id} configuration not found'

# core

def build_dataset(config, core_prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('dataset'))

    return build_core_object2(config, core_prompt, DATASET_MODULE)

def build_model(config, core_prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('model'))

    return build_core_object2(config, core_prompt,
                                MODEL_MODULE, core_suffix=False)

def build_core(config, core_prompt):
    dataset = build_dataset(config.dataset, core_prompt)
    assert not dataset is None
    print('loaded dataset')
    yield dataset

    model = build_model(config.model, core_prompt)
    assert not model is None
    print('loaded model')
    yield model

# learning

def build_data_loaders(config, core_prompt, dataset):
    valid_split = config.validation_split
    if valid_split is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('validation split'))

    class_name = 'data_loader'
    if config[class_name] is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('data loader'))

    data_len = len(dataset)
    indices = [ int(data_len*(1.-valid_split)),
                int(data_len*valid_split) ]
    if indices[0] + indices[1] < data_len: # fix float approx
        indices[0] += data_len - (indices[0]+indices[1])

    train_dataset, valid_dataset = random_split(dataset, indices)
    train_loader =  build_simple_object1(config, core_prompt,
                                            LEARNING_MODULE, class_name,
                                            args=[train_dataset])
    valid_loader =  build_simple_object1(config, core_prompt,
                                            LEARNING_MODULE, class_name,
                                            args=[valid_dataset])   

    return train_loader, valid_loader

def build_early_stopping(config, core_prompt):
    class_name = 'early_stopping'
    if config[class_name] is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('early stopping'))

    return build_simple_object1(config, core_prompt,
                                    LEARNING_MODULE, class_name)

def build_trainer(config, core_prompt, early_stop):
    class_name = 'trainer'
    if config[class_name] is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG(class_name))

    return build_simple_object1(config, core_prompt,
                                    LEARNING_MODULE, class_name,
                                    kwargs={'callbacks':[early_stop]})

def build_loss(config, core_prompt):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('loss'))

    return build_simple_object2(config, core_prompt, LEARNING_MODULE)

def build_optimizer(config, core_prompt, model):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('optimizer'))

    return build_simple_object2(config, core_prompt, LEARNING_MODULE,
                                    args=[model.parameters()])

def build_scheduler(config, core_prompt, optimizer):
    if config is None:
        return None

    return build_simple_object2(config, core_prompt, LEARNING_MODULE,
                                    args=[optimizer])
    
def build_learning(config, core_prompt, dataset, model):
    train_loader, valid_loader = build_data_loaders(config, core_prompt, dataset)
    assert not train_loader is None
    assert not valid_loader is None
    print('loaded data loaders')
    yield (train_loader, valid_loader)

    early_stopping = build_early_stopping(config, core_prompt)
    assert not early_stopping is None
    print('loaded early stopping')

    trainer = build_trainer(config, core_prompt, early_stopping)
    assert not trainer is None
    print('loaded trainer')
    yield trainer

    loss = build_loss(config.loss, core_prompt)
    assert not loss is None
    print('loaded loss')
    yield loss

    optimizer = build_optimizer(config.optimizer, core_prompt, model)
    assert not optimizer is None
    print('loaded optimizer')
    yield optimizer

    scheduler = build_scheduler(config.scheduler, core_prompt, optimizer)
    assert not scheduler is None
    print('loaded scheduler')
    yield scheduler

# logging

def build_tensorboard(config, name):
    config = config.logging
    kwargs = config.tensorboard

    if config.enable:
        kwargs.pop('enable')
        kwargs.name = name
        kwargs.save_dir = os.path.join(LOGS_DIR, TENSORBOARD_DIR)
        logger = TensorBoardLogger(**kwargs)

    return logger

def build_neptune(config):
    config = config.logging
    kwargs = config.neptune

    if config.enable:
        pass

# runner

def run_determinism(config):
    config = config.determinism

    if config.force_determinism:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        seed_everything(config.seed, workers=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    elif not config.seed is None:
        seed_everything(config.seed, workers=config.seed_workers)

def run_core(config):
    core_prompt = build_core_prompt(config)

    core = build_core(config.core, core_prompt)
    dataset = next(core)
    model = next(core)
    print("core section done\n")

    return core_prompt, dataset, model

def run_learning(config, core_prompt, dataset, model):
    learning = build_learning(config.learning, core_prompt,
                                dataset, model)

    train_loader, valid_loader = next(learning)
    trainer = next(learning)
    loss = next(learning)
    optimizer = next(learning)
    scheduler = next(learning)

    model.set_learning(loss, optimizer, scheduler=scheduler)
    model.mount_from_dataset(dataset)
    print('learning section done\n')

    return train_loader, valid_loader, trainer

def run_logging(config):
    # TODO
    pass

def run_fit(trainer, model, train_loader, valid_loader):
    # trainer.fit(model, train_loader, valid_loader)
    pass

INVALID_TASK_MSG = lambda task: f'training task {task} is not supported'

def run_training():
    '''
    Entry point for training executable
    '''
    print("----- TRAINING -----\n")
    config = load_training_config()

    # determinism
    run_determinism(config)
    # core
    prompt, dataset, model = run_core(config)
    # learning
    train, valid, trainer = run_learning(config, prompt, dataset, model)
    # logging
    run_logging(config)
    # fit
    run_fit(trainer, model, train, valid)