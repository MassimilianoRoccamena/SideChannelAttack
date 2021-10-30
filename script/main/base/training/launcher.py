import os
import torch
from datetime import datetime

from main.base.util.string import upper1
from main.base.training.params import EXPERIMENTS_PATH
from main.base.training.params import DATA_MODULE_NAME
from main.base.training.params import MODEL_MODULE_NAME
from main.base.training.config import load_app_config
from main.base.training.reflection import package_name, get_class

def classification_dataset(cfg):
    core_nodes = cfg.problem

    label = cfg.data.label
    class_label = upper1(label)

    class_prefix = upper1(core_nodes[-2])
    class_suffix = upper1(core_nodes[-1])

    class_name = f"{class_label}{class_prefix}{class_suffix}"
    package = package_name(core_nodes)
    
    return get_class(package, DATA_MODULE_NAME, class_name)

def do_training():
    cfg = load_app_config()

    # handling learning types
    core_nodes = cfg.problem
    learning_type = core_nodes[-1]

    if learning_type == "classification":
        dataset_class = classification_dataset(cfg)
        print(dataset_class)
    else:
        raise ValueError("learning type not found")