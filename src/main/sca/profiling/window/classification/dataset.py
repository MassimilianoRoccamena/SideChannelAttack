import torch

from aidenv.api.dlearn.config import build_dataset_kwarg
from aidenv.api.dlearn.dataset import ClassificationDataset
from sca.file.params import str_hex_bytes
from sca.profiling.window.loader import WindowLoader1 as FileConvention1
from sca.profiling.window.loader import WindowLoader2 as FileConvention2
from sca.profiling.window.slicer import StridedSlicer as Strided
from sca.profiling.window.slicer import RandomSlicer as Random
from sca.profiling.window.reader import WindowReader

class WindowClassification(ClassificationDataset):
    '''
    Abstract classification dataset of trace windows.
    '''

    def __init__(self, loader, voltages, frequencies, key_values, trace_indices, channels_first=True):
        '''
        Create new window classification dataset.
        loader: window loader
        voltages: desired voltages
        frequencies: desired frequencies
        key_values: desired key values
        trace_indices: list of trace indices of a file
        channels_first: shape convention of data
        '''
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')

        reader = WindowReader(loader, voltages, frequencies,
                                key_values, trace_indices)
        super().__init__(reader)

        self.channels_first = channels_first

    def data_shape(self):
        return (1,) # only one channel

    @classmethod
    @build_dataset_kwarg('loader')
    def build_kwargs(cls, config, prompt):
        pass

    def tensor_x(self, x):
        '''
        Extends input as a 1 channel sequence.
        '''
        x = torch.Tensor(x)
        if self.channels_first:
            return  x.view(1, *x.size())
        else:
            return  x.view(*x.size(), 1)

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