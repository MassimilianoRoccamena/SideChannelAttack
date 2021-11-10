import os
from omegaconf import OmegaConf

def log_program(config, log_dir):
    OmegaConf.save(config=config, f=os.path.join(log_dir, 'program.yaml'))