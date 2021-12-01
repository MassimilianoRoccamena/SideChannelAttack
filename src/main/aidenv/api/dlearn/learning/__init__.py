from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from aidenv.api.dlearn.learning.logging import LoggableObject
from aidenv.api.dlearn.learning.logging import LoggableScalar
from aidenv.api.dlearn.learning.logging import LoggableLoss
from aidenv.api.dlearn.learning.logging import LoggableTensor
from aidenv.api.dlearn.learning.logging import LoggableFigure
from aidenv.api.dlearn.learning.logging import LoggableAccuracy
from aidenv.api.dlearn.learning.logging import LoggableConfusionMatrix

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        weight = kwargs.get('weight')
        if not weight is None:
            kwargs['weight'] = torch.tensor(weight)
        super().__init__(**kwargs)