import os
import functools
import numpy as np
from collections import defaultdict
from collections.abc import Iterable
from matplotlib.figure import Figure

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerStatus, RunningStage

from main.base.training.scheduler import get_lr_scheduler
from main.base.training.metrics import confusion_matrix_fig
from main.base.training.transforms import compose_transforms
from main.base.training.misc import get_criterion, get_optimizer, is_iterable, fetch_config
from main.core.target import *

class TrainableBase(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.network = self.configure_network()
        self.criterion = self.configure_criterion()

    @property
    @functools.lru_cache()
    def steps_per_epoch(self) -> int:
        if self.trainer.max_steps:
            raise RuntimeError("Training by number of total steps. "
                               "Cannot determine epoch length")

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (min(batches, limit_batches)
                   if isinstance(limit_batches, int)
                   else int(limit_batches * batches))

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return batches // effective_accum

    def configure_criterion(self):
        cfg = self.cfg.criterion
        criterion_cls = get_criterion(cfg.classname)
        criterion_args = cfg.params or {}
        criterion = criterion_cls(**criterion_args)
        return criterion

    def configure_network(self):
        cfg = self.cfg.network
        network_cls = get_model(cfg.classname)
        network = network_cls(**cfg.params)
        return network

    def configure_optimizers(self):
        cfg = self.cfg.optimizer
        optim_cls = get_optimizer(cfg.classname)
        optim = optim_cls(self.network.parameters(),
                          **cfg.params)

        ret_dict = {'optimizer': optim}

        if self.cfg.get('scheduler', None):
            cfg = self.cfg.scheduler
            if 'to_steps_params' in cfg and cfg.to_steps_params:
                for name in cfg.to_steps_params:
                    epoch_value = getattr(cfg.params, name)
                    if is_iterable(epoch_value):
                        for i, val in enumerate(epoch_value):
                            epoch_value[i] = val * self.steps_per_epoch
                    else:
                        iter_value = epoch_value * self.steps_per_epoch
                        setattr(cfg.params, name, iter_value)
            sched_cls = get_lr_scheduler(cfg.classname)
            sched_args = cfg.params or {}

            sched = sched_cls(optim,
                              **sched_args)

            ret_dict['lr_scheduler'] = {
                'scheduler': sched,
                'name': 'learning_rate',
                **cfg.lightning_params
            }

        return ret_dict

    def train_dataloader(self):
        data_cfg = self.cfg.dataset
        dataset_cls = get_dataset(data_cfg.classname)
        dataset_args = {**data_cfg.params, **data_cfg.train.params}
        if fetch_config(self.cfg, "transforms.train"):
            dataset_args['transform'] = compose_transforms(self.cfg.transforms.train)
        train_dataset = dataset_cls(**dataset_args)

        self.classes = train_dataset.classes
        self.num_classes = len(self.classes)

        if fetch_config(data_cfg, 'train.subset', None):
            indices = np.load(data_cfg.train.subset)
            train_dataset = Subset(dataset=train_dataset,
                                   indices=indices)
        load_cfg = self.cfg.dataloader
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  **load_cfg.params)

        return train_loader

    def val_dataloader(self):
        data_cfg = self.cfg.dataset
        dataset_cls = get_dataset(data_cfg.classname)

        if not fetch_config(data_cfg, 'validation.use', True):
            return None

        dataset_args = {**data_cfg.params, **data_cfg.validation.params}
        if fetch_config(self.cfg, "transforms.validation"):
            dataset_args['transform'] = compose_transforms(self.cfg.transforms.validation)
        val_dataset = dataset_cls(**dataset_args)

        if fetch_config(data_cfg, 'validation.subset'):
            indices = np.load(data_cfg.validation.subset)
            val_dataset = Subset(dataset=val_dataset,
                                 indices=indices)

        load_cfg = self.cfg.dataloader
        val_loader = DataLoader(val_dataset,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                **load_cfg.params)
        return val_loader

    def test_dataloader(self):
        data_cfg = self.cfg.dataset
        dataset_cls = get_dataset(data_cfg.classname)
        dataset_args = {**data_cfg.params, **data_cfg.test.params}
        if fetch_config(self.cfg, "transforms.test"):
            dataset_args['transform'] = compose_transforms(self.cfg.transforms.test)
        test_dataset = dataset_cls(**dataset_args)

        load_cfg = self.cfg.dataloader
        test_loader = DataLoader(test_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False,
                                 **load_cfg.params)
        return test_loader

    def log_metric(self, name, value, prog_bar=False, to_file=False, **kwargs):
        self.log(name, value, prog_bar=prog_bar)
        self.logger.log_metrics({name: value}, **kwargs)

        if to_file:
            filename = "metrics_" + name.split("/")[0] + ".txt"
            filepath = os.path.join(self.cfg.log_dir, filename)
            with open(filepath, "a+") as fp:
                fp.write("{} @ {}: {}\n".format(name, self.current_epoch, value))

    def aggregate_eval_metrics(self, outputs, setname):
        all_stats = defaultdict(float)
        tot_samples = sum(o['nsamples'] for o in outputs)

        for output in outputs:
            nsamples = output.pop('nsamples')
            for name, value in output['stats'].items():
                all_stats[name] += value * nsamples / tot_samples

        return {'stats': all_stats}

    def aggregate_fig_metrics(self, outputs, setname, return_fig=False):
        fig_metrics = {}
        if 'conf_matrix' in outputs[0]:
            cm_tot = sum(o['conf_matrix'] for o in outputs)
            if return_fig:
                labels = self.classes if hasattr(self, "classes") else None
                cm_tot = confusion_matrix_fig(cm_tot, labels)
            fig_metrics[f"{setname}/conf_matrix"] = cm_tot

        return fig_metrics

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for name, value in outputs['stats'].items():
            self.log_metric(name, value, step=self.global_step)

    def training_epoch_end(self, outputs):
        aggr_figs = self.aggregate_fig_metrics(outputs, "train", return_fig=True)
        for name, fig in aggr_figs.items():
            self.logger.log_figure(name, fig, step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        if self.trainer.state.status == TrainerStatus.RUNNING and \
                self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return

        aggr_metrics = self.aggregate_eval_metrics(outputs, "validation")
        for name, value in aggr_metrics['stats'].items():
            self.log_metric(name, value, to_file=True, step=self.current_epoch)

        aggr_figs = self.aggregate_fig_metrics(outputs, "validation", return_fig=True)
        for name, fig in aggr_figs.items():
            self.logger.log_figure(name, fig, step=self.current_epoch)

    def test_epoch_end(self, outputs):
        aggr_metrics = self.aggregate_eval_metrics(outputs, "test")
        for name, value in aggr_metrics['stats'].items():
            self.log_metric(name, value, to_file=True, step=self.current_epoch)

        aggr_figs = self.aggregate_fig_metrics(outputs, "test", return_fig=True)
        for name, fig in aggr_figs.items():
            self.logger.log_figure(name, fig, step=self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        return self.evaluation_step(val_batch, batch_idx, "validation")

    def test_step(self, test_batch, batch_idx):
        return self.evaluation_step(test_batch, batch_idx, "test")

    def evaluation_step(self, val_batch, batch_idx, setname):
        pass

    def training_step(self, train_batch, batch_idx):
        pass
