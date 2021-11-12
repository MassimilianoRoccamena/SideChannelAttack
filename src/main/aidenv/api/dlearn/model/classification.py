import itertools
import numpy as np
from textwrap import wrap
from matplotlib import pyplot as plt
from torchmetrics import Accuracy, ConfusionMatrix

from aidenv.api.dlearn.config import build_model_kwarg
from aidenv.api.dlearn.model.logging import LoggableScalar
from aidenv.api.dlearn.model.logging import LoggableFigure
from aidenv.api.dlearn.model.wrapper import WrapperModel
from aidenv.api.dlearn.module.classifier import SingleClassifier
from aidenv.api.dlearn.module.classifier import MultiClassifier

# loggables

class LoggableAccuracy(LoggableScalar):
    '''
    Loggable accuracy value.
    '''

    def __init__(self, *args, **kwargs):
        self.metric = Accuracy(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

class LoggableConfusionMatrix(LoggableFigure):  # matrix aggr not supported
    '''
    Loggable confusion matrix figure.
    '''

    def __init__(self, *args, **kwargs):
        self.metric = ConfusionMatrix(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

    def draw(self, *args, **kwargs):
        matrix = args[0]
        labels = args[1]

        matrix = matrix.astype('float') * 10 / matrix.sum(axis=1)[:, np.newaxis] # normalize
        matrix = np.nan_to_num(matrix, copy=True)
        matrix = matrix.astype('int')

        fig = plt.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(matrix, cmap='Oranges')

        classes = ['\n'.join(wrap(l, 40)) for l in labels]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=4, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            ax.text(j, i, format(matrix[i, j], 'd') if matrix[i, j] != 0 else '.',
                    horizontalalignment='center', fontsize=6,
                    verticalalignment='center', color='black')

        return fig

    def on_epoch_end(self, *args, **kwargs):
        outputs = args[0]
        print(outputs)
        prefix = kwargs.pop('prefix')
        log_name = kwargs.pop('log_name')

        log_name = f'{prefix}/{log_name}'
        matrix = outputs[log_name]
        #labels = self.model.classes if hasattr(self, "classes") else None
        labels = None

        figure = self.draw(matrix, labels)
        super().on_epoch_end(log_name, figure, step=self.model.current_epoch)

# model

class ClassifierModel(WrapperModel):
    '''
    Abstract model wrapping a classifier.
    '''

    def __init__(self, encoder):
        '''
        Create new classifier model.
        encoder: encoder module
        '''
        super().__init__(encoder)

    @classmethod
    @build_model_kwarg('encoder')
    def build_kwargs(cls, config, prompt):
        pass

    def mount(self, *args, **kwargs):
        dataset = args[0]
        labels = dataset.all_labels()

        self.module.set_labels(labels)

        class_logs = { 'accuracy' : LoggableAccuracy() }
                      # 'confusion' : LoggableConfusionMatrix(len(labels)) } # failed aggregation on confusion matrix
        self.add_loggables(class_logs, 'train')
        self.add_loggables(class_logs, 'valid')
        self.add_loggables(class_logs, 'test')

        super().mount(dataset)

class SingleClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a single classifier
    '''

    def __init__(self, encoder):
        super().__init__(SingleClassifier(encoder))

class MultiClassifierModel(ClassifierModel):
    '''
    Abstract model wrapping a multiple classifier
    '''

    def __init__(self, encoder, loggables=None):
        super().__init__(MultiClassifier(encoder))
        raise NotImplementedError