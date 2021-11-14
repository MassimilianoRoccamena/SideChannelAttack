from textwrap import wrap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from torchmetrics import Accuracy, ConfusionMatrix

from aidenv.api.dlearn.model.logging import LoggableScalar
from aidenv.api.dlearn.model.logging import LoggableTensor
from aidenv.api.dlearn.model.logging import LoggableFigure


class LoggableAccuracy(LoggableScalar):
    '''
    Loggable accuracy.
    '''

    def __init__(self, *args, **kwargs):
        self.metric = Accuracy(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

class LoggableConfusionMatrix(LoggableFigure, LoggableTensor):
    '''
    Loggable confusion matrix.
    '''

    def __init__(self, *args, **kwargs):
        self.metric = ConfusionMatrix(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(self.metric(*args))

    def draw(self, *args, **kwargs):
        matrix = args[0]
        labels = args[1]
        epoch = kwargs['epoch']

        matrix = matrix / matrix.sum(axis=0)            # normalize
        matrix = np.nan_to_num(matrix, copy=True)

        labels = ['\n'.join(wrap(l, 10)) for l in labels]

        fig = plt.figure(figsize=(6, 5), facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)

        if not epoch is None:
            title = f'Epoch {epoch}'
        else:
            title = 'Confusion matrix'
        ax.set_title(title, fontsize=14, pad=20)
        
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        sn.heatmap(df, ax=ax, annot=True, cmap='Blues')

        ax.set_xlabel('Truth', fontweight='bold')
        ax.set_ylabel('Prediction', fontweight='bold')
        plt.tight_layout()
        return fig

    def on_epoch_end(self, *args, **kwargs):
        outputs = args[0]
        prefix = kwargs.pop('prefix')
        log_name = kwargs.pop('log_name')
        log_name = f'{prefix}/{log_name}'
        epoch = kwargs.get('epoch')

        tensor = self.build_tensor(outputs, prefix, log_name)

        labels = np.arange(self.shape[0])
        labels = [str(l) for l in labels]
        figure = self.draw(tensor, labels, epoch=epoch)
        log_name = f'{log_name}/figures'

        super().on_epoch_end(log_name, figure, step=self.model.current_epoch)