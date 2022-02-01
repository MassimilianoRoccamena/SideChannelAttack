import os
import itertools
import numpy as np
import pandas as pd
import torch

from textwrap import wrap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from torchmetrics import MetricTracker, Accuracy, Precision, Recall, F1, ConfusionMatrix

# basic

PROGR_BAR_KEY = 'progr_bar'

class LoggableObject:
    '''
    Abstract loggable object of a model.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Create new loggable object.
        '''
        if kwargs.get(PROGR_BAR_KEY) is None:
            progr_bar = False
        else:
            progr_bar = kwargs.pop(PROGR_BAR_KEY)
        self.progr_bar = progr_bar
        self.args = args
        self.kwargs = kwargs

    def mount(self, *args, **kwargs):
        '''
        Post mount the part of the loggable which is function of something.
        dataset: dataset object
        '''
        self.model = args[0]

    def __call__(self, *args, **kwargs):
        '''
        Compute the payload of the loggable object.
        '''
        raise NotImplementedError

class LoggableScalar(LoggableObject):
    '''
    Abstract loggable scalar value.
    '''

    def __init__(self, *args, **kwargs):
        '''
        '''
        super().__init__(*args, **kwargs)
        self.metric = None

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

class LoggableAdvanced(LoggableObject):
    '''
    '''

    def log(self, outputs, log_name, prefix):
        '''
        '''
        raise NotImplementedError

    def on_epoch_end(self, outputs, log_name):
        '''
        '''
        raise NotImplementedError

class LoggableTensor(LoggableAdvanced):
    '''
    Abstract loggable tensorial values.
    '''

    def __call__(self, *args, **kwargs):
        val = args[0]
        self.shape = val.shape
        val = torch.flatten(val)
        return { 'values' : val }

    def log(self, outputs, log_name, prefix):
        logged = self(outputs['prediction'], outputs['target'], prefix=prefix)
        logged_dict = {}

        for k,v in logged.items():
            for i,e in enumerate(v):
                logged_dict[f'{log_name}/{k}/{i}'] = torch.tensor(e, dtype=torch.float32)

        self.model.log_dict(logged_dict, on_step=True, on_epoch=True,
                                sync_dist=True, prog_bar=self.progr_bar)
                
        outputs.update(logged_dict)

    def build_tensor(self, outputs, log_name):
        output = outputs[0]
        log_val_name = f'{log_name}/values'

        shape = self.shape
        tensor = []

        for indices in itertools.product(*[range(s) for s in shape]):
            idx = 0
            if len(shape) < 2:
                idx = indices[0]
            else:
                idx = indices[1] + shape[1]*indices[0]
                if len(shape) > 2:
                    for e,i in enumerate(indices[2:]):
                        idx += shape[:e+2]*i

            tensor.append(output[f'{log_val_name}/{idx}'])
        
        tensor = torch.tensor(tensor)
        tensor = tensor.view(*shape)
        return tensor

class LoggableFigure(LoggableAdvanced):
    '''
    Abstract loggable figure.
    '''

    def on_epoch_end(self, *args, **kwargs):
        self.model.logger.log_figure(*args, **kwargs)

    def draw(self, *args, **kwargs):
        '''
        Draw the figure.
        '''
        raise NotImplementedError

# utils

class LoggableInference(LoggableAdvanced):
    '''
    Loggable model output (prediction, encoding) on some data.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        self.log_encoding = self.kwargs['log_encoding']
        log_dir = os.path.join(self.model.log_dir, 'inference.csv')
        self.file = open(log_dir, 'w')
        self.file_init = True
        self.file_written = False

    def __call__(self, *args, **kwargs):
        if self.file_written:
            return

        target = args[0]
        prediction = args[1]
        encoding = args[2]
        batch_size = prediction.size(dim=0)
        prediction_size = prediction.size(dim=1)
        encoding_size = encoding.size(dim=1)

        if self.file_init:
            header = 'target'
            for i in range(prediction_size):
                header = f'{header},prediction{i}'
            if self.log_encoding:
                for i in range(encoding_size):
                    header = f'{header},encoding{i}'
            print(header, file=self.file)
            self.file_init = False

        for b in range(batch_size):
            label = self.model.labels[target[b]]
            sample_line = f'{label}'

            for i in range(prediction_size):
                sample_line = f'{sample_line},{prediction[b,i]}'
            if self.log_encoding:
                for i in range(encoding_size):
                    sample_line = f'{sample_line},{encoding[b,i]}'
            print(sample_line, file=self.file)

    def log(self, outputs, log_name, prefix):
        target = outputs['target']
        prediction = outputs['prediction']
        encoding = outputs['encoding']
        self(target, prediction, encoding)

    def on_epoch_end(self, *args, **kwargs):
        if not self.file_written:
            self.file.close()
            self.file_written = True

# classification

class LoggableClassifScalar(LoggableScalar):
    '''
    Loggable classification related metric.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        labels = kwargs.get('labels')
        self.kwargs['num_classes'] = len(labels)
        self.metric = self.build_metric()

    def build_metric(self):
        raise NotImplementedError

class LoggableAccuracy(LoggableClassifScalar):
    '''
    Loggable accuracy.
    '''

    def build_metric(self):
        return Accuracy(*self.args, **self.kwargs)

class LoggablePrecision(LoggableClassifScalar):
    '''
    Loggable precision.
    '''

    def build_metric(self):
        return Precision(*self.args, **self.kwargs)

class LoggableRecall(LoggableClassifScalar):
    '''
    Loggable recall.
    '''

    def build_metric(self):
        return Recall(*self.args, **self.kwargs)

class LoggableF1(LoggableClassifScalar):
    '''
    Loggable accuracy.
    '''

    def build_metric(self):
        return F1(*self.args, **self.kwargs)

class LoggableConfusionMatrix(LoggableFigure, LoggableTensor):
    '''
    Loggable confusion matrix.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        labels = kwargs.get('labels')
        self.kwargs['num_classes'] = len(labels)
        self.xticklabels = self.kwargs.pop('xticklabels', 'auto')
        self.yticklabels = self.kwargs.pop('yticklabels', 'auto')
        self.model.conf_mat_train = MetricTracker(ConfusionMatrix(*self.args, **self.kwargs))
        self.model.conf_mat_train.increment()
        self.model.conf_mat_valid = MetricTracker(ConfusionMatrix(*self.args, **self.kwargs))
        self.model.conf_mat_valid.increment()
        self.model.conf_mat_test = MetricTracker(ConfusionMatrix(*self.args, **self.kwargs))
        self.model.conf_mat_test.increment()

    def __call__(self, *args, **kwargs):
        prefix = kwargs['prefix']
        if prefix == 'train':
            cm = self.model.conf_mat_train(*args)
        elif prefix == 'valid':
            cm = self.model.conf_mat_valid(*args)
        elif prefix == 'test':
            cm = self.model.conf_mat_test(*args)
        else:
            raise RuntimeError('encountered invalid prefix')
        
        return super().__call__(cm)

    def draw(self, *args, **kwargs):
        matrix = args[0]
        labels = args[1]
        epoch = kwargs['epoch']

        # matrix = matrix / matrix.sum(axis=0)            # normalize
        matrix = np.nan_to_num(matrix, copy=True)

        labels = ['\n'.join(wrap(l, 10)) for l in labels]

        fig = plt.figure(figsize=(8, 7), facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)

        if not epoch is None:
            title = f'Epoch {epoch}'
        else:
            title = 'Confusion matrix'
        ax.set_title(title, fontsize=14, pad=20)
        
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        sn.heatmap(df, ax=ax, annot=True, cmap='Blues',
                        xticklabels=self.xticklabels, yticklabels=self.yticklabels)

        ax.set_xlabel('Prediction', fontweight='bold', labelpad=10)
        ax.set_ylabel('Truth', fontweight='bold', labelpad=10)
        plt.tight_layout()
        return fig

    def on_epoch_end(self, *args, **kwargs):
        outputs = args[0]
        prefix = kwargs.pop('prefix')
        log_name = kwargs.pop('log_name')
        log_name = f'{prefix}/{log_name}'
        epoch = kwargs.get('epoch')

        tensor = self.build_tensor(outputs, log_name)

        if prefix == 'train':
            cm = self.model.conf_mat_train.compute().cpu()
            self.model.conf_mat_train.reset_all()
        elif prefix == 'valid':
            cm = self.model.conf_mat_valid.compute().cpu()
            self.model.conf_mat_valid.reset_all()
        elif prefix == 'test':
            cm = self.model.conf_mat_test.compute().cpu()
            self.model.conf_mat_test.reset_all()
        else:
            raise RuntimeError('encountered invalid prefix')

        labels = self.model.labels
        figure = self.draw(cm, labels, epoch=epoch)
        log_name = f'{log_name}/figures'

        step = self.model.current_epoch
        super().on_epoch_end(log_name, figure, step=step)
