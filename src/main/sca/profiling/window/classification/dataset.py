import torch

from aidenv.api.dlearn.config import build_dataset_kwarg
from aidenv.api.dlearn.dataset import ClassificationDataset
from sca.file.params import str_hex_bytes
from sca.profiling.window.loader import WindowLoader1 as FileConvention1
from sca.profiling.window.loader import WindowLoader2 as FileConvention2
from sca.profiling.window.reader import WindowReader

class WindowClassification(ClassificationDataset):
    '''
    Abstract classification dataset of trace windows.
    '''

    def __init__(self, loader, data_path, set_name=None, channels_first=True):
        '''
        Create new window classification dataset.
        dataa_path: path of the data folder
        channels_first: shape convention of data
        '''
        reader = WindowReader(loader, data_path, set_name)
        super().__init__(reader)
        self.channels_first = channels_first

    def data_shape(self):
        return (1,) # only one channel

    @classmethod
    @build_dataset_kwarg('loader')
    def build_kwargs(cls, config, prompt):
        pass

    def channels_reshape(self, x):
        '''
        Reshape input as a 1 channel sequence.
        x: pytorch tensor
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
        voltage, frequency, key_value, plain_text, trace_window = self.reader[index]
        x = self.channels_reshape(trace_window)
        labels = self.all_labels()
        label = frequency
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
        voltage, frequency, key_value, plain_text, trace_window = self.reader[index]
        x = self.channels_reshape(trace_window)
        labels = self.all_labels()
        label = (voltage, frequency)
        y0 = labels[0].index(label[0])
        y1 = labels[1].index(label[1])
        return x, (y0, y1)

class VoltageClassification(SingleClassification):
    '''
    Dataset composed of power trace windows labelled with voltage
    '''

    def all_labels(self):
        return self.reader.voltages

class FrequencyClassification(SingleClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    def all_labels(self):
        return self.reader.frequencies