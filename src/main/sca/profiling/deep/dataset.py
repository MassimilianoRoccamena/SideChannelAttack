import math
import numpy as np
import torch

from utils.math import BYTE_SIZE, BYTE_HW_LEN
from aidenv.api.dlearn.config import build_dataset_kwarg
from aidenv.api.dlearn.dataset import ClassificationDataset
from sca.file.params import SBOX_MAT, HAMMING_WEIGHTS
from sca.file.convention1.loader import TraceLoader1 as FileConvention1
from sca.file.convention2.loader import TraceLoader2 as FileConvention2
from sca.file.reader import TraceReader

class TraceClassification(ClassificationDataset):
    '''
    Abstract classification dataset of traces.
    '''

    def __init__(self, loader, voltages, frequencies, key_values, plain_bounds,
                        sets, set_name=None, channels_first=True, trace_len=None):
        '''
        Create new trace classification dataset.
        loader: traces loader
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_bounds: start, end plain text indices
        channels_first: shape convention of data
        trace_len: maximum time index of a trace
        '''
        set_size = None
        for s in sets:
            curr_name = s['name']
            if curr_name == set_name:
                set_size = s['size']
        
        self.plain_bounds = list(plain_bounds)
        num_plains = plain_bounds[1] - plain_bounds[0]
        if set_name == 'test':
            plain_indices = np.arange(int(num_plains-num_plains*set_size), plain_bounds[1])
        else:
            plain_indices = np.arange(plain_bounds[0], int(num_plains*set_size))

        reader = TraceReader(loader, voltages, frequencies, key_values, plain_indices, trace_len=trace_len)
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

class SingleClassification(TraceClassification):
    '''
    Abstract 1-label classification dataset of traces.
    '''

    def __getitem__(self, index):
        voltage, frequency, key_value, plain_idx, \
            trace, plain_text, key = self.reader[index]
        x = self.channels_reshape(trace)
        labels = self.all_labels()
        label = self.current_label(voltage, frequency, key_value, plain_text[0], key)
        y = labels.index(label)
        return x, y

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

class MultiClassification(TraceClassification):
    '''
    Abstract multiple classification dataset of traces.
    '''

    def __getitem__(self, index):
        voltage, frequency, key_value, plain_idx, \
            trace, plain_text, key = self.reader[index]
        x = self.channels_reshape(trace)
        labels = self.all_labels()
        label = self.current_label(voltage, frequency, key_value, plain_text)

        y = []
        for i in range(label):
            y.append(labels[i].index(label[i]))

        return x, y
