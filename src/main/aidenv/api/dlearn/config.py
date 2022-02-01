import os
import torch
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.string import lower_identifier
from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.params import CLASS_NAME_KEY
from aidenv.app.params import DATASET_MODULE
from aidenv.app.params import MODEL_MODULE
from aidenv.app.params import LEARNING_MODULE
from aidenv.app.basic.params import DETERM_SEED_KEY
from aidenv.app.dlearn.params import *
from aidenv.api.config import search_config_key
from aidenv.api.config import load_env_var
from aidenv.api.config import build_object_collapsed
from aidenv.api.config import build_object_expanded
from aidenv.api.dlearn.logging import LoggerCollection

# objects

def build_dataset_object(config, args=None, kwargs=None):
    '''
    Build an expanded dataset object.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_object_expanded(config, DATASET_MODULE, args, kwargs, core_obj=True)

def build_model_object(config, args=None, kwargs=None):
    '''
    Build an expanded model object.
    config: configuration object
    prompt: nodes of the path from the core package
    '''
    return build_object_expanded(config, MODEL_MODULE, args, kwargs, core_obj=True)

def build_learning_object_collapsed(config, class_name, args=None, kwargs=None):
    '''
    Build a collapsed learning object.
    config: configuration object
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    class_name: name of the class
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    '''
    return build_object_collapsed(config, LEARNING_MODULE, class_name, args, kwargs)

def build_learning_object_expanded(config, args=None, kwargs=None):
    '''
    Build an expanded learning object.
    config: configuration object
    prompt: nodes of the path from the core package
    module_name: name of the module file inside the core location
    args: args passed to constructor
    kwargs: kwargs passed to constructor
    '''
    return build_object_expanded(config, LEARNING_MODULE, args, kwargs)

# kwarg

def build_dataset_kwarg(kwarg_name):
    '''
    Decorator for auto-build of a dataset object kwarg
    kwarg_name: parameter name
    '''
    def build_kwarg(cls, config, param):
        obj = build_dataset_object(config[param])
        kwargs = {param:obj}
        config = cls.update_kwargs(config, **kwargs)
        return config

    def decorator(f):
        def wrapper(cls, config):
            return build_kwarg(cls, config, kwarg_name)
        return wrapper
    
    return decorator

def build_model_kwarg(kwarg_name):
    '''
    Decorator for auto-build of a model object kwarg
    kwarg_name: parameter name
    '''
    def build_kwarg(cls, config, param):
        obj = build_model_object(config[param])
        kwargs = {param:obj}
        config = cls.update_kwargs(config, **kwargs)
        return config

    def decorator(f):
        def wrapper(cls, config):
            return build_kwarg(cls, config, kwarg_name)
        return wrapper
    
    return decorator

# determinism

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

# dataset

def build_skip(config):
    skip = search_config_key(config, DATASET_TRAIN_KEY)
    if skip is None:
        return False, False

    train = search_config_key(skip, DATASET_TRAIN_KEY)
    if train is None:
        train = True

    test = search_config_key(skip, DATASET_TEST_KEY)
    if test is None:
        test = True

    return train, test

INVALID_SKIPS_MSG = 'both training and test cannot be skipped'
INVALID_TRAIN_VALID_MSG = 'sum of sizes of train/valid is greater than chunks count'
INVALID_TEST_MSG = 'size of test is 0'
INVALID_SUBSET_MSG = 'dataset subset size type can only be int or float'

def build_dataset(config, hparams):
    skip_train, skip_test = build_skip(config)
    if skip_train and skip_test:
        raise ValueError(INVALID_SKIPS_MSG)

    train_set = None
    valid_set = None
    test_set = None

    if not skip_train:
        train_set = build_dataset_object(config, kwargs={'set_name':'train'})
        valid_set = build_dataset_object(config, kwargs={'set_name':'valid'})
    else:
        print('Training will be skipped')

    if not skip_test:
        test_set = build_dataset_object(config, kwargs={'set_name':'test'})
    else:
        print('Testing will be skipped')

    print('Loaded datasets')
    print(f'Training set has size {len(train_set)}')
    print(f'Validation set has size {len(valid_set)}')
    print(f'Test set has size {len(test_set)}')
    
    dataset_hparams = dict(config)
    hparams.update({DATASET_KEY : dataset_hparams})
    datasets = (train_set, valid_set, test_set)
    return datasets

# model

def build_model(config, hparams):
    model = build_model_object(config)
    print('Loaded model')
    
    checkpoint = search_config_key(config, MODEL_CKPT_KEY)
    if not checkpoint is None:
        print('Loading model checkpoint...')
        model.load_from_checkpoint(checkpoint)
        print('Model checkpoint loaded')

    model_hparams = dict(config)
    hparams.update({MODEL_KEY : model_hparams})
    return model

# learning 1

def build_loss(config,):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_LOSS_KEY))

    return build_learning_object_expanded(config)

def build_early_stopping(config):
    early_stopping = search_config_key(config, LEARN_EARLY_STOP_KEY)
    if early_stopping is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_EARLY_STOP_KEY))

    return build_learning_object_collapsed(config, LEARN_EARLY_STOP_KEY)

def build_optimizer(config, model):
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_OPTIMIZER_KEY))

    return build_learning_object_expanded(config, args=[model.parameters()])

def build_scheduler(config, optimizer):
    if config is None:
        return None

    return build_learning_object_expanded(config, args=[optimizer])

def build_data_loaders(config, datasets):
    data_loader = search_config_key(config, LEARN_DATA_LOAD_KEY)
    if data_loader is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG(LEARN_DATA_LOAD_KEY))

    shuffle = data_loader.shuffle
    train_set, valid_set, test_set = datasets

    # training
    train_loader = None
    if not train_set is None:
        train_loader =  build_learning_object_collapsed(config,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[train_set])

    # validation
    valid_loader = None
    if not valid_set is None:
        data_loader.shuffle = False
        valid_loader =  build_learning_object_collapsed(config,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[valid_set])

    # test
    test_loader = None
    if not test_set is None:
        data_loader.shuffle = False
        test_loader =  build_learning_object_collapsed(config,
                                                LEARN_DATA_LOAD_KEY,
                                                args=[test_set])

    data_loader.shuffle = shuffle
    return train_loader, valid_loader, test_loader
    
def build_learning1(config, hparams, datasets, model):
    loss = search_config_key(config, LEARN_LOSS_KEY)
    loss = build_loss(loss)
    print('Loaded loss')

    early_stop = build_early_stopping(config)
    print('Loaded early stopping')

    optimizer = search_config_key(config, LEARN_OPTIMIZER_KEY)
    optimizer = build_optimizer(optimizer, model)
    print('Loaded optimizer')

    scheduler = search_config_key(config, LEARN_SCHEDULER_KEY)
    scheduler = build_scheduler(scheduler, optimizer)
    print('Loaded scheduler')

    loaders = build_data_loaders(config, datasets)
    print('Loaded data loaders')

    learning_hparams = dict(config)
    del learning_hparams['loggables']
    hparams.update({LEARN_KEY : learning_hparams})
    return early_stop, loss, optimizer, scheduler, loaders

# logging

def build_tensorboard(config, name, log_dir):
    if config is None:
        return None
    
    kwargs = {}
    kwargs.update(dict(config))
    del kwargs[LOG_ENABLE_KEY]
    logger = None

    enable = search_config_key(config, LOG_ENABLE_KEY)
    if enable is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LOG_ENABLE_KEY))

    if config.enable:
        kwargs['name'] = name
        kwargs['save_dir'] = os.path.join(log_dir, TENSORBOARD_DIR)
        logger = TensorBoardLogger(**kwargs)

    return logger

def build_neptune(config, name, id, descr, tags):
    if config is None:
        return None

    kwargs = {}
    kwargs.update(dict(config))
    del kwargs[LOG_ENABLE_KEY]
    logger = None

    enable = search_config_key(config, LOG_ENABLE_KEY)
    if enable is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LOG_ENABLE_KEY))

    if enable:
        user = load_env_var(AIDENV_NEPT_USER_ENV)
        token = load_env_var(AIDENV_NEPT_TOKEN_ENV)
        project = load_env_var(AIDENV_NEPT_PROJECT_ENV)

        kwargs['prefix'] = 'experiment'
        kwargs['project'] = f'{user}/{project}'
        kwargs['name'] = f'{name}_{id}'
        kwargs['custom_run_id'] = id
        kwargs['description'] = descr
        kwargs['tags'] = tags + list(kwargs['tags'])

        logger = NeptuneLogger(api_key=token, **kwargs)
    
    return logger

def build_logging(config, hparams, origin, name, id, log_dir, descr):
    # tensorboard
    tensorboard = search_config_key(config, LOG_TB_KEY)
    tensorboard = build_tensorboard(tensorboard, name, log_dir)
    if not tensorboard is None:
        tensorboard.log_hyperparams(hparams)
        print('Loaded tensorboard logger')

    # neptune
    tag = ''
    for o in origin:
        tag = f'{tag}-{o}'
    tags = [tag]            # default origin tag

    neptune = search_config_key(config, LOG_NEPT_KEY)
    neptune = build_neptune(neptune, name, id, descr, tags)
    if not neptune is None:
        neptune.log_hyperparams(hparams)
        print('Loaded neptune logger')

    output = []
    if not tensorboard is None:
        output.append(tensorboard)
    if not neptune is None:
        output.append(neptune)

    if len(output) == 0:
        raise ValueError('You must specify at least one logger')

    return tuple(output)

# learning 2

def build_trainer(config, callbacks, loggers, log_dir):
    trainer = search_config_key(config, LEARN_TRAINER_KEY)
    if trainer is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_TRAINER_KEY))

    loggers = LoggerCollection(loggers)

    return build_learning_object_collapsed(config,
                                    LEARN_TRAINER_KEY,
                                    kwargs= {
                                        'logger' : loggers,
                                        'callbacks' : callbacks,
                                        'default_root_dir' : log_dir
                                    })

#def build_metrics(config):
#    if config is None:
#        return None
#    metrics = {}
#    for c in config:
#        cls_name = c[CLASS_NAME_KEY]
#        log_name = lower_identifier(cls_name)
#        metrics[log_name] = build_learning_object_expanded(c)
#    return metrics

def build_loggables(config):
    if config is None:
        return None

    loggables = {}
    sets = {}
    for c in config:
        cls_name = c[CLASS_NAME_KEY]
        c[CLASS_NAME_KEY] = f'Loggable{cls_name}'

        log_name = lower_identifier(cls_name)
        loggables[log_name] = build_learning_object_expanded(c)
        ss = search_config_key(c, LEARN_SETS_KEY)
        if ss is None:
            ss = ['train','valid','test']
        else:
            ss = list(ss)
        sets[log_name] = ss

    return loggables, sets

def build_learning2(config, early_stop, loggers, log_dir):
    ckpt_path = os.path.join(log_dir, CHECKPOINT_DIR)
    ckpt_callback = ModelCheckpoint(dirpath=ckpt_path, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [ early_stop, ckpt_callback, lr_monitor ]
    trainer = build_trainer(config, callbacks, loggers, log_dir)
    print('Loaded trainer')

    #metrics = search_config_key(config, LEARN_METRICS_KEY)
    #metrics = build_metrics(metrics)
    #print('Loaded metrics')

    loggables = search_config_key(config, LEARN_LOG_KEY)
    loggables, sets = build_loggables(loggables)
    print('Loaded loggables')

    return trainer, loggables, sets