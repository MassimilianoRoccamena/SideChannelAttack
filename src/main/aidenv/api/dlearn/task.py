from aidenv.api.task import CoreTask
from aidenv.app.params import CONFIG_NOT_FOUND_MSG
from aidenv.api.config import get_program_config
from aidenv.api.config import get_program_origin
from aidenv.api.config import get_program_name
from aidenv.api.config import get_program_id
from aidenv.api.config import get_program_log_dir
from aidenv.api.config import get_program_description
from aidenv.api.config import search_config_key
from aidenv.app.dlearn.params import DATASET_KEY
from aidenv.app.dlearn.params import MODEL_KEY
from aidenv.app.dlearn.params import LEARN_KEY
from aidenv.app.dlearn.params import LOG_KEY
from aidenv.api.dlearn.config import build_dataset
from aidenv.api.dlearn.config import build_model
from aidenv.api.dlearn.config import build_learning1
from aidenv.api.dlearn.config import build_logging
from aidenv.api.dlearn.config import build_learning2

class  DeepLearningTask(CoreTask):
    '''
    Abstract machine learning task.
    '''

    pass

class DeepClassification(DeepLearningTask):
    '''
    Deep learning classification task
    '''

    def parse_dataset(self, config, hparams):
        config = search_config_key(config, DATASET_KEY)
        if config is None:
            raise KeyError(CONFIG_NOT_FOUND_MSG(DATASET_KEY))

        datasets = build_dataset(config, hparams)

        print('Datasets configuration done')
        return datasets
    
    def parse_model(self, config, hparams):
        config = search_config_key(config, MODEL_KEY)
        if config is None:
            raise KeyError(CONFIG_NOT_FOUND_MSG(MODEL_KEY))

        model = build_model(config, hparams)

        print('Model configuration done')
        return model

    def parse_learning1(self, config, hparams, datasets, model):
        config = search_config_key(config, LEARN_KEY)
        if config is None:
            raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_KEY))

        early_stop, loss, optimizer, scheduler, loaders = \
                            build_learning1(config, hparams, datasets, model)

        model.set_learning(loss, optimizer, scheduler=scheduler)

        print('Learning 1 configuration done')
        return early_stop, loaders

    def parse_logging(self, config, hparams, origin, name, id, log_dir, descr):
        config = search_config_key(config, LOG_KEY)
        if config is None:
            raise KeyError(CONFIG_NOT_FOUND_MSG(LOG_KEY))

        loggers = build_logging(config, hparams, origin,
                                    name, id, log_dir, descr)

        print("Logging configuration done")
        return loggers

    def parse_learning2(self, config, datasets, model, early_stop, loggers, log_dir):
        config = search_config_key(config, LEARN_KEY)
        if config is None:
            raise KeyError(CONFIG_NOT_FOUND_MSG(LEARN_KEY))

        trainer, loggables, sets = build_learning2(config, early_stop,
                                                            loggers, log_dir)

        model.add_loggables(loggables, sets)
        model.mount(datasets[0], log_dir)

        print('Learning 2 configuration done')
        return trainer

    def run_train_test(self, trainer, model, train_loader, valid_loader, test_loader):
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

    def run(self, *args):
        config = get_program_config()
        origin = get_program_origin()
        name = get_program_name()
        id = get_program_id()
        log_dir = get_program_log_dir()
        descr = get_program_description()
        hparams = {}

        datasets = self.parse_dataset(config, hparams)
        model = self.parse_model(config, hparams)
        early_stop, loaders = self.parse_learning1(config, hparams,
                                                    datasets, model)
        loggers = self.parse_logging(config, hparams, origin,
                                        name, id, log_dir, descr)
        trainer = self.parse_learning2(config, datasets, model,
                                        early_stop, loggers, log_dir)

        self.run_train_test(trainer, model, *loaders)