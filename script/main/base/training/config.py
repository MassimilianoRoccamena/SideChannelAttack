from omegaconf import OmegaConf

APP_CONFIG_PATH = 'config/training.yaml'

def load_config(config_path):
    return OmegaConf.load(config_path)

def load_app_config():
    return load_config(APP_CONFIG_PATH)