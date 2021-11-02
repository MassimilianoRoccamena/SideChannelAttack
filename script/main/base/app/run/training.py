import os

from main.base.app.params import CONFIG_DIR
from main.base.app.params import DATASET_MODULE
from main.base.app.params import MODEL_MODULE
from main.base.app.config import load_config
from main.base.app.config import config_object, config_core_object

TRAINING_CONF_PATH = os.path.join(CONFIG_DIR, 'training.yaml')

def load_training_config():
    return load_config(TRAINING_CONF_PATH)

def config_dataset(config, core_nodes):
    if config is None:
        raise ValueError('dataset configuration not found')

    return config_core_object(config, core_nodes, DATASET_MODULE)

def config_model(config, core_nodes):
    if config is None:
        raise ValueError('model configuration not found')

    return config_core_object(config, core_nodes,
                                MODEL_MODULE, core_suffix=False)

def run_training():
    config = load_training_config()
    core_nodes = config.core

    dataset = config_dataset(config.dataset, core_nodes)
    assert not dataset is None

    model = config_model(config.model, core_nodes)
    assert not model is None