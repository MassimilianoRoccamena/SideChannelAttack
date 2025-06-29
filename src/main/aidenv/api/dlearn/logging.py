from pytorch_lightning.loggers.base import LoggerCollection as LoggerCollection_
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

class LoggerCollection(LoggerCollection_):
    @property
    def name(self):
        return self._logger_iterable[0].name

    @property
    def version(self):
        return self._logger_iterable[0].version

    def log_figure(self, log_name: str, fig: Figure, step: Optional[int] = None,
                   close: bool = True):
        for logger in self._logger_iterable:
            if isinstance(logger, NeptuneLogger):
                logger.log_image(log_name, fig, step)
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure(log_name, fig, step)
            else:
                raise RuntimeError("Failed to log figure")

        if close:
            plt.close(fig)
