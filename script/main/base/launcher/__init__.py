import os
import torch
from datetime import datetime

from main.base.util.string import upper1
from main.base.launcher.params import EXPERIMENTS_PATH
from main.base.launcher.params import DATASET_MODULE_NAME
from main.base.launcher.params import MODEL_MODULE_NAME
from main.base.launcher.config import load_app_config
from main.base.launcher.reflection import package_name, get_class

def parse_window_slicer(cfg):
    pass

def parse_classification_dataset(cfg):
    core_nodes = cfg.core
    package = package_name(core_nodes)

    label = cfg.dataset.label
    class_label = upper1(label)

    class_prefix = core_nodes[-2]
    is_window = class_prefix == 'window'
    class_prefix = upper1(core_nodes[-2])
    class_suffix = upper1(core_nodes[-1])

    # window case
    if is_window:
        parse_window_slicer(cfg.dataset.params.slicer)

    class_name = f"{class_label}{class_prefix}{class_suffix}"
    
    return get_class(package, DATASET_MODULE_NAME, class_name)

def parse_model(cfg):
    core_nodes = cfg.core
    package = package_name(core_nodes)

    class_name = cfg.model.classname

    return get_class(package, MODEL_MODULE_NAME, class_name)

def do_training():
    cfg = load_app_config()
    core_nodes = cfg.core

    # dataset
    learning_type = core_nodes[-1]

    if learning_type == "classification":
        dataset = parse_classification_dataset(cfg)
        if dataset is None:
            raise ValueError('invalid dataset specified')
    else:
        raise ValueError('learning type not found')

    # model
    model = parse_model(cfg)
    if model is None:
            raise ValueError('invalid model specified')