from main.mlenv.app.deepgym.config import CONFIG_NOT_FOUND_MSG
from main.mlenv.app.deepgym.config import load_training_config
from main.mlenv.app.deepgym.config import build_base
from main.mlenv.app.deepgym.config import build_determinism
from main.mlenv.app.deepgym.config import build_logging
from main.mlenv.app.deepgym.config import build_core
from main.mlenv.app.deepgym.config import build_learning

# sections parsers

def parse_base(config):
    config = config.base
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('base'))

    prompt, name, log_dir = build_base(config)
    
    print('Base configuration done')
    return prompt, name, log_dir

def parse_determinism(config):
    config = config.determinism
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('determinism'))

    build_determinism(config)

    print('Determinism configuration done')

def parse_logging(config, name, log_dir):
    config = config.logging
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('logging'))

    loggers = build_logging(config, name, log_dir)

    print("Logging configuration done")
    return loggers

def parse_core(config, prompt):
    config = config.core
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('core'))

    dataset, model = build_core(config, prompt)

    print("Core configuration done")
    return dataset, model

def parse_learning(config, prompt, dataset, model, loggers, log_dir):
    config = config.learning
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('learning'))

    learning = build_learning(config, prompt,
                                dataset, model, loggers, log_dir)

    loaders = next(learning)
    trainer = next(learning)
    loss = next(learning)
    optimizer = next(learning)
    scheduler = next(learning)

    model.set_learning(loss, optimizer, scheduler=scheduler)
    model.mount_from_dataset(dataset)

    print('Learning configuration done')
    return loaders, trainer

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

def run():
    '''
    Entry point for deep-gym executable
    '''
    print("Deep gym started")

    config = load_training_config()
    prompt, name, log_dir = parse_base(config)

    parse_determinism(config)
    loggers = parse_logging(config, name, log_dir)

    dataset, model = parse_core(config, prompt)
    loaders, trainer = parse_learning(config, prompt, dataset,
                                            model, loggers, log_dir)

    run_train_test(trainer, model, *loaders) # test WIP

    print('Deep gym finished')