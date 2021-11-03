import os

from main.base.app.params import CONFIG_DIR
from main.base.app.params import DATASET_MODULE
from main.base.app.params import MODEL_MODULE
from main.base.app.params import LEARNING_MODULE
from main.base.app.config import load_config, config_core_prompt
from main.base.app.config import config_simple_object1, config_simple_object2
from main.base.app.config import config_core_object2

# basic

TRAINING_CONF_PATH = os.path.join(CONFIG_DIR, 'training.yaml')

def load_training_config():
    return load_config(TRAINING_CONF_PATH)

CONFIG_NOT_FOUND_MSG = lambda id: f'{id} configuration not found'

# core

def config_dataset(config, core_nodes):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('dataset'))

    return config_core_object2(config, core_nodes, DATASET_MODULE)

def config_model(config, core_nodes):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('model'))

    return config_core_object2(config, core_nodes,
                                MODEL_MODULE, core_suffix=False)

def config_core(config, core_nodes):
    dataset = config_dataset(config.dataset, core_nodes)
    assert not dataset is None
    model = config_model(config.model, core_nodes)
    assert not model is None

    return dataset, None

# learning

def config_data_loader(config, core_nodes, dataset):
    class_name = 'data_loader'
    if config[class_name] is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('data loader'))

    return config_simple_object1(config, core_nodes,
                                    LEARNING_MODULE, class_name, args=[dataset])

def config_loss(config, core_nodes):
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('loss'))

    return config_simple_object2(config, core_nodes, LEARNING_MODULE)
    
def config_learning(config, core_nodes, dataset):
    loss = config_loss(config.loss, core_nodes)
    assert not loss is None
    data_loader = config_data_loader(config, core_nodes, dataset)
    assert not data_loader is None

    return loss, data_loader

# runner

def run_training():
    config = load_training_config()
    core_nodes = config_core_prompt(config)

    core = config_core(config.core, core_nodes)
    dataset = core[0]
    model = core[1]

    learning = config_learning(config.learning, core_nodes, dataset)
    loss = learning[0]
    data_loader = learning[1]

    model.set_learning(loss, None)

    if core_nodes[1] == 'classification':
        model.mount_classifier(dataset)