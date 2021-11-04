import os
from torch.utils.data import random_split

from main.base.app.params import CONFIG_DIR
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
    model = build_model(config.model, core_prompt)
    assert not model is None
    print('loaded model')

    return dataset, model

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

# determinism

# logging

# runner

INVALID_TASK_MSG = lambda task: f'training task {task} is not supported'

def run_training():
    config = load_training_config()

    # determinism
    # TODO

    # core
    print("----- TRAINING -----\n")
    core_prompt = build_core_prompt(config)
    core = build_core(config.core, core_prompt)
    dataset = core[0]
    model = core[1]
    print("core section done\n")

    # learning
    learning = build_learning(config.learning, core_prompt,
                                dataset, model)
    train_loader, valid_loader = next(learning)
    trainer = next(learning)
    loss = next(learning)
    optimizer = next(learning)
    scheduler = next(learning)
    print('learning section done')

    model.set_learning(loss, optimizer, scheduler=scheduler)
    model.mount_from_dataset(dataset)

    # logging
    # TODO

    # fit
    # trainer.fit(model, train_loader, valid_loader)