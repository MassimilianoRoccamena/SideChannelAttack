from math import ceil
import numpy as np

from aidenv.api.config import CoreObject
from sca.file.params import TRACE_SIZE

class TraceSlicer(CoreObject):
    '''
    Abstract core slicer of traces into windows.
    '''

    INVALID_INDEX_MSG = 'invalid trace window index'

    def __init__(self, window_size, num_windows, trace_len=None):
        '''
        Create new slicer of traces into windows.
        window_size: temporal window size
        num_windows: number of windows in a trace
        trace_len: max trace length
        '''
        self.window_size = window_size
        self.num_windows = num_windows
        if trace_len is None:
            self.trace_len = TRACE_SIZE
        else:
            if trace_len > TRACE_SIZE:
                raise ValueError(f'trace length {max_start} is too big')
            else:
                self.trace_len = trace_len

    def validate_window_index(self, window_index):
        '''
        Check consistency of a window index.
        window_index: window index of a trace
        '''
        if window_index < 0 or window_index >= self.num_windows:
            raise IndexError(TraceSlicer.INVALID_INDEX_MSG)

    def slice(self, window_index):
        '''
        Compute start, end indices of a given trace window.
        window_index: window index of a trace
        '''
        raise NotImplementedError

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        return self.slice(index)

class RandomSlicer(TraceSlicer):
    '''
    Trace window random slicer.
    '''

    def slice(self, window_index):
        if window_index < 0:
            window_index = self.num_windows - window_index

        self.validate_window_index(window_index)

        start = np.random.randint(0, high=self.trace_len-self.window_size)
        return start, start+self.window_size-1

class StridedSlicer(TraceSlicer):
    '''
    Trace window slicer with striding.
    '''

    def __init__(self, window_size, stride, trace_len=None):
        '''
        Create new strided slicer of traces.
        window_size: temporal window size
        stride: spacing between adjacent windows
        trace_len: max trace length
        '''
        self.stride = stride
        max_idx = self.trace_len - window_size + 1
        nwindows = ceil((max_idx+1) / stride) + 1

        super().__init__(window_size, nwindows, trace_len=trace_len)

    def slice(self, window_index):
        if window_index < 0:
            window_index = self.num_windows - window_index

        self.validate_window_index(window_index)

        start = window_index * self.stride
        if (start+self.window_size-1 >= self.trace_len):
            start = self.trace_len - self.window_size

        return start, start+self.window_size-1
