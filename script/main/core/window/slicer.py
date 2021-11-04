from math import ceil

from main.base.data.params import TRACE_SIZE
from main.base.app.config import CoreObject

class TraceSlicer(CoreObject):
    '''
    Abstract core slicer of traces into windows.
    '''

    INVALID_INDEX_MSG = 'invalid trace window index'

    def __init__(self, window_size, num_windows):
        '''
        Create new slicer of traces into windows.
        window_size: temporal window size
        num_windows: number of windows in a trace
        '''
        self.window_size = window_size
        self.num_windows = num_windows

    @classmethod
    def build_args(cls, config, core_prompt):
        return [ config.window_size ]

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

class StridedSlicer(TraceSlicer):
    '''
    Trace window slicer with striding.
    '''

    def __init__(self, window_size, stride):
        '''
        Create new strided slicer of traces.
        window_size: temporal window size
        stride: spacing between adjacent windows
        '''
        self.stride = stride
        max_idx = TRACE_SIZE - window_size + 1
        nwindows = ceil((max_idx+1) / stride) + 1

        super().__init__(window_size, nwindows)

    @classmethod
    def build_args(cls, config, core_prompt):
        new_args =  [ config.stride ]
        return cls.build_super_args(config, core_prompt) + new_args

    def slice(self, window_index):
        if window_index < 0:
            window_index = self.num_windows - window_index

        self.validate_window_index(window_index)

        start = window_index * self.stride
        if (start+self.window_size-1 >= TRACE_SIZE):
            start = TRACE_SIZE - self.window_size

        return start, start+self.window_size-1
