import os
import errno
import functools

import numpy as np
from sklearn.model_selection import train_test_split

from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from pathlib import Path
from collections import MutableMapping, Iterable
from omegaconf.basecontainer import BaseContainer

import torch
from torch import nn
from torch import optim
from torch.nn.modules.loss import _Loss

class FunctionReprWrapper:
    def __init__(self, repr, func):
        self._repr = repr
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kw):
        return self._func(*args, **kw)

    def __repr__(self):
        return self._repr

def withrepr(reprfun):
    def _wrap(func):
        return FunctionReprWrapper(reprfun, func)
    return _wrap

def is_iterable(iter):
    try:
        if len(iter) > 0:
            _ = iter[0]
        return True
    except:
        return False

def flatten_config(cfg, parent_key='', sep='.'):
    items = []
    if isinstance(cfg, BaseContainer):
        if isinstance(cfg, MutableMapping):
            for k, v in cfg.items():
                new_key = parent_key + sep + k if parent_key else k
                items.extend(flatten_config(v, new_key, sep=sep).items())
        elif is_iterable(cfg) and not isinstance(cfg, str):
            if len(cfg) > 0 and isinstance(cfg[0], BaseContainer):
                for i, v in enumerate(cfg):
                    new_key = parent_key + sep + str(i) if parent_key else str(i)
                    items.extend(flatten_config(v, new_key, sep=sep).items())
            else:
                return {parent_key: list(cfg)}
        return dict(items)
    elif callable(cfg) is True:
        return {parent_key: str(cfg)}
    else:
        return {parent_key: cfg}

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_source_files(extensions):
    if extensions:
        source_files = [str(path) for ext in extensions
                        for path in Path('./').rglob(ext)]
        return source_files
    else:
        return None

def fetch_config(cfg, name, default=None):
    names = name.split(".")
    try:
        for name in names[:-1]:
            cfg = getattr(cfg, name)
        return getattr(cfg, names[-1])
    except ConfigAttributeError:
        return default