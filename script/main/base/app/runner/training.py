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

def config_dataset(config, core_nodes):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('dataset'))

    return build_core_object2(config, core_nodes, DATASET_MODULE)

def config_model(config, core_nodes):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('model'))

    return build_core_object2(config, core_nodes,
                                MODEL_MODULE, core_suffix=False)

def config_core(config, core_nodes):
    dataset = config_dataset(config.dataset, core_nodes)
    assert not dataset is None
    print('loaded dataset')
    model = config_model(config.model, core_nodes)
    assert not model is None
    print('loaded model')

    return dataset, model

# learning

def config_data_loaders(config, core_nodes, dataset):
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
    train_loader =  build_simple_object1(config, core_nodes,
                                            LEARNING_MODULE, class_name,
                                            args=[train_dataset])
    valid_loader =  build_simple_object1(config, core_nodes,
                                            LEARNING_MODULE, class_name,
                                            args=[valid_dataset])   

    return train_loader, valid_loader

def config_early_stopping(config, core_nodes):
    class_name = 'early_stopping'
    if config[class_name] is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('early stopping'))

    return build_simple_object1(config, core_nodes,
                                    LEARNING_MODULE, class_name)

def config_trainer(config, core_nodes, early_stop):
    class_name = 'trainer'
    if config[class_name] is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG(class_name))

    return build_simple_object1(config, core_nodes,
                                    LEARNING_MODULE, class_name,
                                    kwargs={'callbacks':[early_stop]})

def config_loss(config, core_nodes):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('loss'))

    return build_simple_object2(config, core_nodes, LEARNING_MODULE)

def config_optimizer(config, core_nodes, model):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('optimizer'))

    return build_simple_object2(config, core_nodes, LEARNING_MODULE,
                                    args=[model.parameters()])

def config_scheduler(config, core_nodes, optimizer):
    if config is None:
        return None

    return build_simple_object2(config, core_nodes, LEARNING_MODULE,
                                    args=[optimizer])
    
def config_learning(config, core_nodes, dataset, model):
    train_loader, valid_loader = config_data_loaders(config, core_nodes, dataset)
    assert not train_loader is None
    assert not valid_loader is None
    print('loaded data loaders')
    yield (train_loader, valid_loader)

    early_stopping = config_early_stopping(config, core_nodes)
    assert not early_stopping is None
    print('loaded early stopping')

    trainer = config_trainer(config, core_nodes, early_stopping)
    assert not trainer is None
    print('loaded trainer')
    yield trainer

    loss = config_loss(config.loss, core_nodes)
    assert not loss is None
    print('loaded loss')
    yield loss

    optimizer = config_optimizer(config.optimizer, core_nodes, model)
    assert not optimizer is None
    print('loaded optimizer')
    yield optimizer

    scheduler = config_scheduler(config.scheduler, core_nodes, optimizer)
    assert not scheduler is None
    print('loaded scheduler')
    yield scheduler

# determinism

# logging

# runner

INVALID_TASK_MSG = lambda task: f'training task {task} is not supported'

def handle_training_task(task, dataset, model):
    if task == 'classification':
        model.mount_classifier(dataset)
    else:
        raise ValueError(INVALID_TASK_MSG(task))

def run_training():
    config = load_training_config()

    # determinism
    # TODO

    # core
    print("----- TRAINING -----\n")
    core_nodes = build_core_prompt(config)
    core = config_core(config.core, core_nodes)
    dataset = core[0]
    model = core[1]
    print("core section done\n")

    # learning
    learning = config_learning(config.learning, core_nodes,
                                dataset, model)
    train_loader, valid_loader = next(learning)
    trainer = next(learning)
    loss = next(learning)
    optimizer = next(learning)
    scheduler = next(learning)
    print('learning section done')

    model.set_learning(loss, optimizer, scheduler=scheduler)
    handle_training_task(core_nodes[1], dataset, model)

    # logging
    # TODO

    # fit
    # trainer.fit(model, train_loader, valid_loader)