import os
import torch
from pytorch_lightning.utilities.seed import seed_everything

from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.basic.params import DETERM_SEED_KEY
from aidenv.app.dlearn.params import *
from aidenv.app.config import search_config_key

def build_determinism(config):
    force = search_config_key(config, DETERM_FORCE_KEY)
    if force is None:
        force = False
        print('Not forcing determinism')

    seed = search_config_key(config, DETERM_SEED_KEY)
    if seed is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_SEED_KEY))

    if force:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        seed_everything(seed, workers=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        workers = search_config_key(config, DETERM_WORKERS_KEY)
        if workers is None:
            workers = True
            print('Seeding workers')

        seed_everything(seed, workers=workers)