import torch
import torch.nn.functional as F

import numpy as np
from main.base.training.trainable.base import TrainableBase
from main.base.training.metrics import accuracy, Timer
from sklearn.metrics import confusion_matrix

class TrainableClassification(TrainableBase):

    def training_step(self, train_batch, batch_idx):
        xyz = train_batch['xyz']
        target = train_batch['target']
        features = train_batch.get('features', None)

        with Timer() as fw_timer:
            output = self.network(xyz, features)

        loss = self.criterion(output, target)
        prob = F.softmax(input=output, dim=1)
        pred = torch.argmax(prob, dim=1)

        labels = np.arange(self.num_classes) if hasattr(self, "classes") else None
        cm = confusion_matrix(target.detach().cpu().numpy(),
                              pred.detach().cpu().numpy(),
                              labels=labels)
        acc1, acc3, acc5 = accuracy(output, target, topk=(1, 3, 5))

        return {
            'loss': loss,
            'stats': {
                'train/loss': loss.item(),
                'train/acc1': acc1.item(),
                'train/acc3': acc3.item(),
                'train/acc5': acc5.item(),
                'train/fw_time': fw_timer.time
            },
            'conf_matrix': cm
        }

    def evaluation_step(self, val_batch, batch_idx, setname):
        xyz = val_batch['xyz']
        target = val_batch['target']
        features = val_batch.get('features', None)
        batch_size = xyz.shape[0]

        with Timer() as fw_timer:
            output = self.network(xyz, features)

        loss = self.criterion(output, target)
        prob = F.softmax(input=output, dim=1)
        pred = torch.argmax(prob, dim=1)

        labels = np.arange(self.num_classes) if hasattr(self, "classes") else None
        cm = confusion_matrix(target.detach().cpu().numpy(),
                              pred.detach().cpu().numpy(),
                              labels=labels)
        acc1, acc3, acc5 = accuracy(output, target, topk=(1, 3, 5))

        return {
            'stats': {
                f'{setname}/loss': loss.item(),
                f'{setname}/acc1': acc1.item(),
                f'{setname}/acc3': acc3.item(),
                f'{setname}/acc5': acc5.item(),
                f'{setname}/fw_time': fw_timer.time
            },
            'nsamples': batch_size,
            'conf_matrix': cm
        }
