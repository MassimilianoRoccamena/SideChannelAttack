from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping