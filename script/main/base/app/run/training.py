from main.base.app.params import DATASET_MODULE
from main.base.app.params import MODEL_MODULE
from main.base.app.config import load_training_config
from main.base.app.config import config_object, config_core_object

def run_training():
    config = load_training_config()
    core_nodes = config.core

    config_dataset = config.dataset
    if config_dataset is None:
        raise ValueError('dataset configuration not found')

    dataset = config_core_object(config_dataset, core_nodes, DATASET_MODULE)
    assert not dataset is None

    config_model = config.model
    if config_model is None:
        raise ValueError('model configuration not found')

    model = config_core_object(config_model, core_nodes,
                                MODEL_MODULE, core_suffix=False)
    assert not model is None