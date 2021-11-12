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
from aidenv.app.dlearn.config import build_logging
from aidenv.app.dlearn.config import build_core
from aidenv.app.dlearn.config import build_learning

# sections parsers

def parse_base(config):
    config = search_config_key(config, BASE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(BASE_KEY))

    prompt, name, log_dir = build_base(config)
    
    print('Base configuration done')
    return prompt, name, log_dir

def parse_determinism(config):
    config = search_config_key(config, DETERM_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(DETERM_KEY))

    build_determinism(config)

    print('Determinism configuration done')

def parse_logging(config, name, log_dir):
    config = search_config_key(config, LOG_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LOG_KEY))

    loggers = build_logging(config, name, log_dir)

    print("Logging configuration done")
    return loggers

def parse_core(config, prompt):
    config = search_config_key(config, CORE_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(CORE_KEY))

    dataset, nsamples, model = build_core(config, prompt)

    print("Core configuration done")
    return dataset, nsamples, model

def parse_learning(config, prompt, dataset, nsamples, model, loggers, log_dir):
    config = search_config_key(config, LEARN_KEY)
    if config is None:
        raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_KEY))

    learning = build_learning(config, prompt,
                                dataset, nsamples,
                                model, loggers, log_dir)

    trainer = next(learning)
    loss = next(learning)
    optimizer = next(learning)
    scheduler = next(learning)

    model.set_learning(loss, optimizer, scheduler=scheduler)
    model.mount(dataset)

    loaders = next(learning)

    print('Learning configuration done')
    return trainer, loaders

# main runners

def run_train_test(trainer, model, train_loader, valid_loader, test_loader):
    if not (train_loader is None or valid_loader is None):
        trainer.fit(model, train_loader, valid_loader)
        print('Model training done')
    else:
        print('Model training skipped')

    if not test_loader is None:
        trainer.test(model, test_loader)
        print('Model testing done')
    else:
        print('Model testing skipped')

def run(*args):
    '''
    Entry point for dlearn environment
    args: program arguments
    '''
    print("Deep learning environment started")

    config = get_program_config()

    prompt, name, log_dir = parse_base(config)
    parse_determinism(config)
    loggers = parse_logging(config, name, log_dir)

    dataset, nsamples, model = parse_core(config, prompt)
    trainer, loaders = parse_learning(config, prompt, dataset, nsamples,
                                            model, loggers, log_dir)

    run_train_test(trainer, model, *loaders)

    print('Deep learning environment finished')