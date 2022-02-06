import torch

from utils.math import BYTE_SIZE, BYTE_HW_LEN
from aidenv.api.dlearn.config import build_dataset_kwarg
from aidenv.api.dlearn.dataset import ClassificationDataset
from sca.file.params import SBOX_MAT, HAMMING_WEIGHTS
from sca.profiling.aligner.window.loader import WindowLoader1 as FileConvention1
from sca.profiling.aligner.window.loader import WindowLoader2 as FileConvention2
from sca.profiling.aligner.window.reader import WindowReader

class WindowClassification(ClassificationDataset):
    '''
    Abstract classification dataset of trace windows.
    '''

    def __init__(self, loader, lookup_path, set_name=None, channels_first=True):
        '''
        Create new window classification dataset.
        lookup_path: path of the lookup data folder
        channels_first: shape convention of data
        '''
        reader = WindowReader(loader, lookup_path, set_name)
        super().__init__(reader)
        self.channels_first = channels_first

    def data_shape(self):
        return (1,) # only one channel

    @classmethod
    @build_dataset_kwarg('loader')
    def build_kwargs(cls, config):
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
        label = self.current_label(voltage, frequency, key_value, plain_text)
        y = labels.index(label)
        return x, y

class VoltageClassification(SingleClassification):
    '''
    Dataset composed of power trace windows labelled with voltage
    '''

    def current_label(self, *args):
        return args[0]

    def all_labels(self):
        return self.reader.voltages

class FrequencyClassification(SingleClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    def current_label(self, *args):
        return args[1]

    def all_labels(self):
        return self.reader.frequencies

class KeyClassification(SingleClassification):
    '''
    Dataset composed of power traces labelled with key value.
    '''

    def current_label(self, *args):
        return args[2]

    def all_labels(self):
        return self.reader.key_values

class HammingKeyClassification(SingleClassification):
    '''
    Dataset composed of power traces labelled with hamming weight
    of key value.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_hw = math.ceil(math.log(len(self.reader.key_values),2))
        self.hamming_wights = [str(i) for i in range(max_hw+1)]

    def current_label(self, *args):
        key = args[2]
        key_idx = self.reader.key_values.index(key)
        hw_key = HAMMING_WEIGHTS[key_idx]
        return str(hw_key)

    def all_labels(self):
        return self.hamming_wights

class HammingSboxClassification(SingleClassification):
    '''
    Dataset composed of power traces labelled with hamming weight
    of sbox output.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hamming_wights = [str(i) for i in range(BYTE_HW_LEN)]

    def current_label(self, *args):
        plain_text = args[3]
        key = args[4]
        hw_sbox = HAMMING_WEIGHTS[SBOX_MAT[plain_text[0] ^ key[0]]]
        return str(hw_sbox)

    def all_labels(self):
        return self.hamming_wights

class MultiClassification(WindowClassification):
    '''
    Abstract multiple classification dataset of trace windows.
    '''

    def __getitem__(self, index):
        voltage, frequency, key_value, plain_text, trace_window = self.reader[index]
        x = self.channels_reshape(trace_window)
        labels = self.all_labels()
        label = self.current_label(voltage, frequency, key_value, plain_text)

        y = []
        for i in range(label):
            y.append(labels[i].index(label[i]))

        return x, y
