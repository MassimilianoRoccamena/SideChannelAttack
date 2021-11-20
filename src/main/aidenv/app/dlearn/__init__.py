from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.app.config import get_program_config
from aidenv.app.config import search_config_key
from aidenv.app.dlearn.params import BASE_KEY
from aidenv.app.dlearn.params import DETERM_KEY
from aidenv.app.dlearn.params import LOG_KEY
from aidenv.app.dlearn.params import CORE_KEY
from aidenv.app.dlearn.params import LEARN_KEY
from aidenv.app.dlearn.config import build_base
from aidenv.app.dlearn.config import build_determinism
from aidenv.app.dlearn.config import build_core
from aidenv.app.dlearn.config import build_learning1
from aidenv.app.dlearn.config import build_logging
from aidenv.app.dlearn.config import build_learning2

# sections parsers

def parse_base(config):
    config = search_config_key(config, BASE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_KEY))

    origin, prompt, name, id, log_dir, descr = build_base(config)
    
    print('Base configuration done')
    return origin, prompt, name, id, log_dir, descr

def parse_determinism(config):
    config = search_config_key(config, DETERM_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_KEY))

    build_determinism(config)

    print('Determinism configuration done')

def parse_core(config, hparams, prompt):
    config = search_config_key(config, CORE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(CORE_KEY))

    datasets, subset_size, model = build_core(config, hparams, prompt)

    print("Core configuration done")
    return datasets, subset_size, model

def parse_learning1(config, hparams, prompt, datasets, subset_size, model):
    config = search_config_key(config, LEARN_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_KEY))

    early_stop, loss, optimizer, scheduler, loaders = \
                        build_learning1(config, hparams, prompt,
                                        datasets, subset_size, model)

    model.set_learning(loss, optimizer, scheduler=scheduler)

    print('Learning 1 configuration done')
    return early_stop, loaders

def parse_logging(config, hparams, origin, prompt, name, id, log_dir, descr):
    config = search_config_key(config, LOG_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LOG_KEY))

    loggers = build_logging(config, hparams, origin, prompt,
                                name, id, log_dir, descr)

    print("Logging configuration done")
    return loggers

def parse_learning2(config, prompt, datasets, model, early_stop, loggers, log_dir):
    config = search_config_key(config, LEARN_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_KEY))

    trainer, loggables = build_learning2(config, prompt, early_stop,
                                loggers, log_dir)

    model.add_loggables(loggables, 'train')
    model.add_loggables(loggables, 'valid')
    model.add_loggables(loggables, 'test')

    model.mount(datasets[0])

    print('Learning 2 configuration done')
    return trainer

# main runners

def run_train_test(trainer, model, train_loader, valid_loader, test_loader):
    if not (train_loader is None or valid_loader is None):
        print('')
        trainer.fit(model, train_loader, valid_loader)
        print('')
        print('Model training done')
    else:
        print('Model training skipped')

    if not test_loader is None:
        print('')
        trainer.test(model, test_loader)
        print('')
        print('Model testing done')
    else:
        print('Model testing skipped')

def run(*args):
    '''
    Entry point for dlearn environment
    args: program arguments
    '''
    print(' ---=== AIDENV ===--- \n')
    print("Deep learning environment started")
    config = get_program_config()
    hparams = {}

    origin, prompt, name, id, log_dir, descr = parse_base(config)
    parse_determinism(config)
    datasets, subset_size, model = parse_core(config, hparams, prompt)
    early_stop, loaders = parse_learning1(config, hparams, prompt,
                                            datasets, subset_size, model)
    loggers = parse_logging(config, hparams, origin, prompt,
                                name, id, log_dir, descr)
    trainer = parse_learning2(config, prompt, datasets, model,
                                early_stop, loggers, log_dir)

    run_train_test(trainer, model, *loaders)
    print('Deep learning environment finished')