import os
import torch
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import training # what is this?
from src.main.base.training.args import parse_args
from src.main.base.training.misc import fetch_config, flatten_config, get_source_files
from src.main.base.training.logs import LoggerCollection, HyperParamsLogger

def do_training():
    cfg = parse_args()

    if fetch_config(cfg, "force_determinism", False):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        seed_everything(cfg.seed, workers=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    elif fetch_config(cfg, "seed", None) is not None:
        seed_everything(cfg.seed, workers=cfg.get('seed_workers', True))

    trainable_class = getattr(training, cfg.trainable.classname)
    trainable = trainable_class(cfg)

    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.log_dir = os.path.join("runs", cfg.exp_name, dt_string)
    params = flatten_config(cfg)

    tensorboard_args = dict(cfg.loggers.tensorboard)
    tensorboard_args.pop("enable")
    tensorboard_args["save_dir"] = tensorboard_args.pop("save_dir", "runs")
    tb_logger = TensorBoardLogger(
        name=cfg.exp_name,
        version=dt_string,
        **tensorboard_args
    )
    tb_logger.log_hyperparams(params)
    loggers = [tb_logger]

    if cfg.loggers.neptune and cfg.loggers.neptune.enable:
        neptune_args = dict(cfg.loggers.neptune)
        upload_source_files = get_source_files(neptune_args['upload_source_files'])
        neptune_args.pop("upload_source_files")
        neptune_args.pop("enable")

        loggers.append(NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            upload_source_files=upload_source_files,
            params=params,
            **neptune_args
        ))

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    params_writer = HyperParamsLogger(cfg)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)

    early_stopping = None
    if fetch_config(cfg, "early_stopping.params"):
        early_stopping = EarlyStopping(**cfg.early_stopping.params)

    trainer = pl.Trainer(logger=LoggerCollection(loggers),
                         callbacks=[lr_monitor, checkpoint_callback,
                                    params_writer, early_stopping],
                         default_root_dir="runs",
                         **cfg.trainer)

    if cfg.load_checkpoint:
        print('loading checkpoint...')

        trainable = trainable.load_from_checkpoint(cfg.load_checkpoint, cfg=cfg)

    if not cfg.test_only:
        trainer.fit(trainable)
    trainer.test(trainable)
