import torch

from main.mlenv.api.deep.config import build_dataset_object1
from main.mlenv.api.deep.dataset import ClassificationDataset
from main.sca.core.window.loader import WindowLoader1 as Convention1
from main.sca.core.window.slicer import StridedSlicer as Strided
from main.sca.core.window.reader import WindowReader

class WindowClassification(ClassificationDataset):
    '''
    Abstract classification dataset of trace windows.
    '''

    def __init__(self, loader, voltages, frequencies, key_values, num_traces):
        '''
        Create new window classification dataset.
        loader: window loader
        voltages: desired voltages
        frequencies: desired frequencies
        key_values: desired key values
        num_traces: number of traces in each file
        '''
        reader = WindowReader(loader, voltages, frequencies,
                                key_values, num_traces)
        super().__init__(reader)

    @classmethod
    def build_kwargs(cls, config, prompt):
        loader = build_dataset_object1(config.loader, prompt)
        config = cls.update_kwargs(config, loader=loader)
        return config

    def tensor_x(self, x):
        '''
        Extends input as a 1 channel sequence.
        '''
        x = torch.Tensor(x)
        return  x.view(1, *x.size())

class SingleClassification(WindowClassification):
    '''
    Abstract 1-label classification dataset of trace windows.
    '''

    def __getitem__(self, index):
        x = self.reader[index]
        x = self.tensor_x(x)
        labels = self.all_labels()
        label = self.current_label()
        y = labels.index(label)
        return x, y

class MultiClassification(WindowClassification):
    '''
    Abstract (volt,freq)-classification dataset of trace windows.
    '''

    def all_labels(self):
        return ( self.reader.voltages,
                 self.reader.frequencies )

    def current_label(self):
        return ( self.reader.file_id.voltage,
                 self.reader.file_id.frequency )

    def __getitem__(self, index):
        x = self.reader[index]
        x = self.tensor_x(x)
        labels = self.all_labels()
        label = self.current_label()
        y0 = labels[0].index(label[0])
        y1 = labels[1].index(label[1])
        return x, (y0, y1)

class VoltageClassification(SingleClassification):
    '''
    Dataset composed of power trace windows labelled with voltage
    '''

    def all_labels(self):
        return self.reader.voltages

    def current_label(self):
        return self.reader.file_id.voltage

class FrequencyClassification(SingleClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    def all_labels(self):
        return self.reader.frequencies

    def current_label(self):
        return self.reader.file_id.frequency