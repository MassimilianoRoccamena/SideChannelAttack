from main.mlenv.app.deepgym.config import CONFIG_NOT_FOUND_MSG
from main.mlenv.app.deepgym.config import load_training_config
from main.mlenv.app.deepgym.config import build_base
from main.mlenv.app.deepgym.config import build_determinism
from main.mlenv.app.deepgym.config import build_logging
from main.mlenv.app.deepgym.config import build_core
from main.mlenv.app.deepgym.config import build_learning

# main sections parsers

def parse_base(config):
    config = config.base
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('base'))

    prompt, name, log_dir, skip = build_base(config)
    
    print('base configuration done')
    return prompt, name, log_dir, skip

def parse_determinism(config):
    config = config.determinism
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('determinism'))

    build_determinism(config)

    print('determinism configuration done')

def parse_logging(config, name, log_dir):
    config = config.logging
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('logging'))

    loggers = build_logging(config, name, log_dir)

    print("logging configuration done")
    return loggers

def parse_core(config, prompt):
    config = config.core
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('core'))

    dataset, model = build_core(config, prompt)

    print("core configuration done")
    return dataset, model

def parse_learning(config, prompt, dataset, model, loggers, log_dir):
    config = config.learning
    if config is None:
        raise ValueError(CONFIG_NOT_FOUND_MSG('learning'))

    learning = build_learning(config, prompt,
                                dataset, model, loggers, log_dir)

    train_loader, valid_loader = next(learning)
    trainer = next(learning)
    loss = next(learning)
    optimizer = next(learning)
    scheduler = next(learning)

    model.set_learning(loss, optimizer, scheduler=scheduler)
    model.mount_from_dataset(dataset)

    print('learning configuration done')
    return train_loader, valid_loader, trainer

# main runners

def run_train_test(trainer, model, train_loader, valid_loader, test_loader):
    if not (train_loader is None or valid_loader is None):
        trainer.fit(model, train_loader, valid_loader)
        print('model training done')
    else:
        print('model training skipped')

    if not test_loader is None:
        trainer.test(model, test_loader)
        print('model testing done')
    else:
        print('model testing skipped')

def run():
    '''
    Entry point for deep-gym executable
    '''
    print("deep gym started")

    config = load_training_config()
    prompt, name, log_dir, skip = parse_base(config)

    if skip['training'] and skip['testing']:
        print('found nothing to do')
    else:
        parse_determinism(config)
        loggers = parse_logging(config, name, log_dir)

        dataset, model = parse_core(config, prompt)
        train, valid, trainer = parse_learning(config, prompt, dataset,
                                                model, loggers, log_dir)

        run_train_test(trainer, model, train, valid, None) # test WIP
    
    print('deep gym finished')