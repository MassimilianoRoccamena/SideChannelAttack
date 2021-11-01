from main.base.launcher.params import DATASET_MODULE
from main.base.launcher.params import MODEL_MODULE
from main.base.launcher.config import load_training_config
from main.base.launcher.config import parse_object, parse_core_object

def launch_training():
    config = load_training_config()
    core_nodes = config.core

    config_dataset = config.dataset
    if config_dataset is None:
        raise ValueError('dataset configuration not found')

    dataset = parse_core_object(config_dataset, core_nodes, DATASET_MODULE)
    assert not dataset is None

    config_model = config.model
    if config_model is None:
        raise ValueError('model configuration not found')

    model = parse_core_object(config_model, core_nodes,
                                MODEL_MODULE, core_suffix=False)
    assert not model is None