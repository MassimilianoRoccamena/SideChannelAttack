from math import ceil

from core.data.params import TRACE_SIZE

class BasicTraceSlicer:
    '''
    Abstract slicer of a trace into fixed size subwindows.
    '''
    INVALID_INDEX_MSG = "invalid trace window index"

    def __init__(self, window_size, num_windows):
        '''
        Create new slicer of traces into windows.
        window_size: temporal window size
        num_windows: number of windows in a trace
        '''
        self.window_size = window_size
        self.num_windows = num_windows

    def validate_window_index(self, index):
        '''
        Check consistency of a window index.
        indedx: window index
        '''
        if index < 0 or index >= self.num_windows:
            raise IndexError(BasicTraceSlicer.INVALID_INDEX_MSG)

    def window_bounds(self, index):
        '''
        Compute start, end indices of a given trace window.
        index: window index
        '''
        raise NotImplementedError

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        return self.window_bounds(index)

class AdvancedTraceSlicer(BasicTraceSlicer):
    '''
    Trace windows slicer with striding.
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

    def window_bounds(self, index):
        if index < 0:
            index = self.num_windows - index

        self.validate_window_index(index)

        start = index * self.stride
        if (start+self.window_size-1 >= TRACE_SIZE):
            start = TRACE_SIZE - self.window_size

        return start, start+self.window_size-1