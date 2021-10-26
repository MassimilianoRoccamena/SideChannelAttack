from math import ceil
import numpy as np

from core.data.params import TRACE_LENGTH

class TraceSlicer:
    '''
    Abstract slicer of a trace into fixed size subwindows
    '''

    def __init__(self, nwindows):
        self.nwindows = nwindows

    def validate_index(self, idx):
        '''
        Check consistency of a window index
        idx: window index
        '''
        if idx < 0 or idx > self.nwindows:
            raise Exception("invalid trace window index")

    def window_bounds(self, idx):
        '''
        Compute start, end indices of a given trace window
        idx: window index
        '''
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.window_bounds(idx)

class StridedSlicer(TraceSlicer):
    '''
    Trace windows slicer with stride and shuffling
    '''

    def __init__(self, window, stride, shuffle):
        self.window = window
        self.stride = stride

        max_idx = TRACE_LENGTH - window + 1
        nwindows = ceil((max_idx+1) / stride) + 1
        super().__init__(nwindows)

        self.shuffle = shuffle
        self.shuffle_indices()

    def shuffle_indices(self):
        '''
        Shuffle indices of trace windows
        '''
        if self.shuffle is True:
            self.shuffle_idx = np.random.randint(self.nwindows, size=self.nwindows)

    def window_bounds(self, idx):
        if idx < 0:
            idx = self.nwindows - idx

        self.validate_index(idx)

        if self.shuffle is True:
            idx = self.shuffle_idx[idx]

        start = idx * self.stride
        if (start+self.window-1 >= TRACE_LENGTH):
            start = TRACE_LENGTH - self.window

        return start, start+self.window-1

class SequentialSlicer(StridedSlicer):
    '''
    Slicer of a trace into sequential windows
    '''

    def __init__(self, window):
        super().__init__(window, 1, False)

class RandomSlicer(StridedSlicer):
    '''
    Slicer of a trace into random windows
    '''

    def __init__(self, window):
        super().__init__(window, 1, True)