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
from sca.assembler import DynamicTraceReader

class TraceClassification(ClassificationDataset):
    '''
    Abstract classification dataset of traces.
    '''

    def __init__(self, voltages, frequencies, key_values, plain_bounds,
                        sets, set_name=None, channels_first=True):
        '''
        Create new trace classification dataset.
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_bounds: start, end plain text indices
        sets: partitioning over all data
        set_name: name of the object partition
        channels_first: shape convention of data
        '''
        set_size = None
        for s in sets:
            curr_name = s['name']
            if curr_name == set_name:
                set_size = s['size']
        
        self.plain_bounds = list(plain_bounds)
        num_plains = plain_bounds[1] - plain_bounds[0]
        if set_name == 'test':
            self.plain_indices = np.arange(int(num_plains-num_plains*set_size), plain_bounds[1])
        else:
            self.plain_indices = np.arange(plain_bounds[0], int(num_plains*set_size))

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

class StaticTraceClassification(TraceClassification):
    '''
    '''

    def __init__(self, loader, voltages, frequencies, key_values, plain_bounds,
                        sets, set_name=None, channels_first=True):
        '''
        Create new trace classification dataset.
        loader: traces loader
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_bounds: start, end plain text indices
        sets: partitioning over all data
        set_name: name of the object partition
        channels_first: shape convention of data
        '''
        super().__init__(voltages, frequencies, key_values, plain_bounds,
                            sets, set_name=set_name, channels_first=channels_first)
        reader = TraceReader(loader, voltages, frequencies, key_values, self.plain_indices)
        ClassificationDataset.__init__(self, reader)

class DynamicTraceClassification(TraceClassification):
    '''
    '''

    def __init__(self, loader, voltages, frequencies, key_values, plain_bounds,
                        num_dynamics, mu, sigma, min_window_len, max_window_len,
                        sets, set_name=None, channels_first=True):
        '''
        Create new trace classification dataset.
        loader: traces loader
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_bounds: start, end plain text indices
        num_dynamics: number of dynamic traces for some given key_value, plain_text
        mu: mean of the gaussian duration of the static windows
        sigma: std of the gaussian duration of the static windows
        min_window_len: min static window size
        max_window_len: max static window size
        sets: partitioning over all data
        set_name: name of the object partition
        channels_first: shape convention of data
        '''
        super().__init__(voltages, frequencies, key_values, plain_bounds,
                            sets, set_name=set_name, channels_first=channels_first)
        reader = DynamicTraceReader(loader, voltages[0], frequencies, key_values, self.plain_indices,
                                        num_dynamics, mu, sigma, min_window_len, max_window_len)
        ClassificationDataset.__init__(self, reader)

class SingleLabel:
    '''
    Abstract 1-label getitem wrapper.
    '''

    def __getitem__(self, index):
        voltage, frequency, key_value, plain_idx, \
            trace, plain_text, key = self.reader[index]
        x = self.channels_reshape(trace)
        labels = self.all_labels()
        label = self.current_label(voltage, frequency, key_value, plain_text[0], key)
        y = labels.index(label)
        return x, y

class KeyLabel(SingleLabel):
    '''
    Abstract key labelling methods container.
    '''

    def current_label(self, *args):
        return args[2]

    def all_labels(self):
        return self.reader.key_values

class KeyStatic(KeyLabel, StaticTraceClassification):
    '''
    Key classification dataset of static traces.
    '''

    pass

class KeyDynamic(KeyLabel, DynamicTraceClassification):
    '''
    Key classification dataset of dynamic traces.
    '''

    pass

class HammingKeyLabel(SingleLabel):
    '''
    Abstract key hamming weight labelling methods container.
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

class HammingKeyStatic(HammingKeyLabel, StaticTraceClassification):
    '''
    Key  hamming weightclassification dataset of static traces.
    '''

    pass

class HammingKeyDynamic(HammingKeyLabel, DynamicTraceClassification):
    '''
    Key hamming weight classification dataset of static traces.
    '''

    pass

class HammingSboxLabel(SingleLabel):
    '''
    Abstract sbox output hamming weight labelling methods container.
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

class HammingSboxStatic(HammingSboxLabel, StaticTraceClassification):
    '''
    Sbox output hamming weight classification dataset of static traces.
    '''

    pass

class HammingSboxDynamic(HammingSboxLabel, DynamicTraceClassification):
    '''
    Sbox output hamming weight classification dataset of static traces.
    '''

    pass

class MultiLabel:
    '''
    Abstract n-label getitem wrapper.
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
