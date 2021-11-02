import os

from main.base.util.string import upper1
from main.base.app.params import CONFIG_DIR
from main.base.app.params import DATASET_MODULE
from main.base.app.params import MODEL_MODULE
from main.base.app.params import LEARNING_MODULE
from main.base.app.config import load_config, config_core_prompt
from main.base.app.config import config_object, config_object1
from main.base.app.config import config_object2, config_object3

TRAINING_CONF_PATH = os.path.join(CONFIG_DIR, 'training.yaml')

# loading

def load_training_config():
    return load_config(TRAINING_CONF_PATH)

# core

def config_core_object1(config, core_nodes, module_name):
    return config_object3(lambda cls: cls.from_config(config.params, core_nodes),
                            core_nodes, module_name, config)

def config_core_object2(config, core_nodes, module_name, core_suffix=True):
    class_prefix = upper1(core_nodes[-1])
    if core_suffix:
        class_suffix = upper1(core_nodes[-2])
    else:
        class_suffix = config.name
    class_name = f"{class_prefix}{class_suffix}"
    return config_object(lambda cls: cls.from_config(config.params, core_nodes),
                            core_nodes[:-1], module_name, class_name)

def config_dataset(config, core_nodes):
    if config is None:
        raise ValueError('dataset configuration not found')

    return config_core_object2(config, core_nodes, DATASET_MODULE)

def config_model(config, core_nodes):
    if config is None:
        raise ValueError('model configuration not found')

    return config_core_object2(config, core_nodes,
                                MODEL_MODULE, core_suffix=False)

def config_core(config, core_nodes):
    dataset = config_dataset(config.dataset, core_nodes)
    assert not dataset is None
    #model = config_model(config.model, core_nodes)     # invalid tensorboard on local conda
    #assert not model is None

    return dataset, None

# learning

def config_learning_object1(config, core_nodes, module_name, class_name, args=[]):
    return config_object1(lambda cls: cls(*args, **config[class_name]), core_nodes[:-1],
                            module_name, config, class_name)

def config_learning_object2():
    pass

def config_data_loader(config, core_nodes, dataset):
    class_name = 'data_loader'
    if config[class_name] is None:
        raise ValueError('data loader configuration not found')

    return config_learning_object1(config, core_nodes,
                                    LEARNING_MODULE, class_name, args=[dataset])

def config_learning(config, core_nodes, dataset):
    data_loader = config_data_loader(config, core_nodes, dataset)
    assert not data_loader is None

    return data_loader

# runner

def run_training():
    config = load_training_config()
    core_nodes = config_core_prompt(config)

    core = config_core(config.core, core_nodes)
    dataset = core[0]
    model = core[1]

    learning = config_learning(config.learning, core_nodes, dataset)
    data_loader = learning
    #TODO