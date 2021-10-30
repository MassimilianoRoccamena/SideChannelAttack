import os
import torch
from datetime import datetime

from main.base.training.config import load_app_config
from main.base.training.reflection import package_name, get_class

EXPERIMENTS_PATH = ".log"

DATA_MODULE_NAME = "dataset"
MODEL_MODULE_NAME = "model"

def to_upper_first(s):
    prefix = s[0].upper()
    suffix = s[1:]
    return f"{prefix}{suffix}"

def dataset_classification(cfg):
    core_nodes = cfg.problem

    label = cfg.data.label
    class_label = to_upper_first(label)

    class_prefix = to_upper_first(core_nodes[-2])
    class_suffix = to_upper_first(core_nodes[-1])

    class_name = f"{class_label}{class_prefix}{class_suffix}"
    package = package_name(core_nodes)
    
    _class = get_class(package, DATA_MODULE_NAME, class_name)

def do_training():
    cfg = load_app_config()

    # handling learning types
    core_nodes = cfg.problem
    learning_type = core_nodes[-1]

    if learning_type == "classification":
        dataset_class = dataset_classification(cfg)
        print(dataset_class)
    else:
        raise ValueError("learning type not found")