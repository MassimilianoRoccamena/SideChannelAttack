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
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix

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
        Compute the payload of the object.
        '''
        raise NotImplementedError
    
    def log(self, outputs, log_name, *step_results):
        '''
        Log the payload of the object.
        '''
        raise NotImplementedError

    def on_epoch_end(self, *args, **kwargs):
        '''
        Handle model training epoch end.
        '''
        raise NotImplementedError

class LoggableScalar(LoggableObject):
    '''
    Abstract loggable scalar.
    '''

    def log(self, outputs, log_name, *step_results):
        loss = step_results[0]
        target = step_results[1]
        prediction = step_results[2]
        payload = self(prediction, target, loss=loss)

        self.model.log(log_name, payload, on_step=True, on_epoch=True,
                        sync_dist=True, prog_bar=self.progr_bar)
        outputs[log_name] = payload

    def on_epoch_end(self, *args, **kwargs):
        pass

class LoggableTensor(LoggableObject):
    '''
    Abstract loggable tensor.
    '''

    def __call__(self, *args, **kwargs):
        val = args[0]
        self.shape = val.shape
        return { 'values' : torch.flatten(val) }

    def log(self, outputs, log_name, *step_results):
        loss = step_results[0]
        target = step_results[1]
        prediction = step_results[2]
        log = self(prediction, target, loss=loss)
        payload = {}

        for k,v in log.items():
            for i,e in enumerate(v):
                payload[f'{log_name}/{k}/{i}'] = e

        self.model.log_dict(payload, on_step=True, on_epoch=True,
                                sync_dist=True, prog_bar=self.progr_bar)
                
        outputs.update(payload)

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

class LoggableFigure(LoggableObject):
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

# metrics

class LoggableLoss(LoggableScalar):
    '''
    Loggable loss.
    '''

    def __call__(self, *args, **kwargs):
        return kwargs['loss']

class LoggableAccuracy(LoggableScalar):
    '''
    Loggable accuracy.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        self.metric = Accuracy(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

class LoggablePrecision(LoggableScalar):
    '''
    Loggable precision.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        self.metric = Precision(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

class LoggableRecall(LoggableScalar):
    '''
    Loggable recall.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        self.metric = Recall(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

class LoggableF1(LoggableScalar):
    '''
    Loggable accuracy.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        self.metric = F1(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args)

class LoggableConfusionMatrix(LoggableFigure, LoggableTensor):
    '''
    Loggable confusion matrix.
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        num_classes = len(kwargs['labels'])
        self.metric = ConfusionMatrix(num_classes,
                                        *self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(self.metric(*args))

    def draw(self, *args, **kwargs):
        matrix = args[0]
        labels = args[1]
        epoch = kwargs['epoch']

        # matrix = matrix / matrix.sum(axis=0)            # normalize
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

        #labels = np.arange(self.shape[0])
        #labels = [str(l) for l in labels]
        labels = self.model.labels
        figure = self.draw(tensor, labels, epoch=epoch)
        log_name = f'{log_name}/figures'

        step = 0
        step = self.model.current_epoch

        super().on_epoch_end(log_name, figure, step=step)

class LoggableInference(LoggableObject):
    '''
    Loggable model inference (on test set).
    '''

    def mount(self, *args, **kwargs):
        super().mount(*args, **kwargs)
        self.log_encoding = self.kwargs['log_encoding']
        log_dir = os.path.join(self.model.log_dir, 'inference.csv')
        self.file = open(log_dir, 'w')
        self.file_init = True

    def __call__(self, *args, **kwargs):
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

    def log(self, outputs, log_name, *step_results):
        loss = step_results[0]
        target = step_results[1]
        prediction = step_results[2]
        encoding = step_results[3]
        self(target, prediction, encoding)

    def on_epoch_end(self, *args, **kwargs):
        self.file.close()