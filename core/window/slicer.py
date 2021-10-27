from math import ceil
import numpy as np

from core.data.params import TRACE_SIZE

class BasicTraceSlicer:
    '''
    Abstract slicer of a trace into fixed size subwindows
    '''
    INVALID_INDEX_MSG = "invalid trace window index"

    def __init__(self, wsize, nwindows):
        '''
        Create new slicer of windows
        wsize: window size
        nwindows: number of windows in a trace
        '''
        self.window_size = wsize
        self.nwindows = nwindows

    def validate_index(self, idx):
        '''
        Check consistency of a window index
        idx: window index
        '''
        if idx < 0 or idx >= self.nwindows:
            raise IndexError(BasicTraceSlicer.INVALID_INDEX_MSG)

    def window_bounds(self, idx):
        '''
        Compute start, end indices of a given trace window
        idx: window index
        '''
        raise NotImplementedError

    def __len__(self):
        return self.nwindows

    def __getitem__(self, idx):
        return self.window_bounds(idx)

class StridedTraceSlicer(BasicTraceSlicer):
    '''
    Trace windows slicer with stride and shuffling
    '''

    def __init__(self, wsize, stride, shuffle):
        self.stride = stride
        max_idx = TRACE_SIZE - wsize + 1
        nwindows = ceil((max_idx+1) / stride) + 1

        super().__init__(wsize, nwindows)

        self.shuffle = shuffle
        self.shuffle_indices()

    def shuffle_indices(self):
        '''
        Shuffle indices of trace windows
        '''
        if self.shuffle:
            self.windows_idx = np.random.randint(self.nwindows, size=self.nwindows)

    def window_bounds(self, idx):
        if idx < 0:
            idx = self.nwindows - idx

        self.validate_index(idx)

        if self.shuffle:
            idx = self.windows_idx[idx]

        start = idx * self.stride
        if (start+self.window_size-1 >= TRACE_SIZE):
            start = TRACE_SIZE - self.window_size

        return start, start+self.window_size-1

class SequentialTraceSlicer(StridedTraceSlicer):
    '''
    Slicer of a trace into sequential windows
    '''

    def __init__(self, wsize):
        super().__init__(wsize, 1, False)

class RandomTraceSlicer(StridedTraceSlicer):
    '''
    Slicer of a trace into random windows
    '''

    def __init__(self, wsize):
        super().__init__(wsize, 1, True)