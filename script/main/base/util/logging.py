import os
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

class HyperParamsLogger(Callback):
    def __init__(self, config):
        self.config = config

    def on_train_start(self, trainer, pl_module):
        OmegaConf.save(config=self.config, f=os.path.join(self.config.log_dir, "params.yaml"))