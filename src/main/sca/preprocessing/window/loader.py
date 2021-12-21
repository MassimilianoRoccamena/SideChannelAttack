import numpy as np

from aidenv.api.dprocess.config import build_task_kwarg
from sca.file.convention1.loader import TraceLoader1
from sca.file.convention2.loader import TraceLoader2

class WindowLoader:
    '''
    Loader from raw file of power trace windows.
    '''

    def __init__(self, slicer):
        '''
        Create new loader of trace windows.
        slicer: trace slicer
        '''
        self.slicer = slicer
        self.window_start = None
        self.window_end = None

    @classmethod
    @build_task_kwarg('slicer')
    def build_kwargs(cls, config, prompt):
        pass

    def load_window_of_some_traces(self, plain_indices, window_index):
        '''
        Load a window (and corresponding plain text) of some traces from a raw file.
        plain_indices: plain text indices of a file
        window_index: window index of a trace
        '''
        start, end = self.slicer[window_index]
        self.window_start = start
        self.window_end = end
        time_idx = np.arange(start, end+1)
        traces, plain_texts = self.load_some_projected_traces(plain_indices, time_idx)
        return traces[0], plain_texts[0]

class WindowLoader1(WindowLoader, TraceLoader1):
    '''
    Loader from raw file of power trace windows using filesystem convention 1.
    '''

    pass

class WindowLoader2(WindowLoader, TraceLoader2):
    '''
    Loader from raw file of power trace windows using filesystem convention 2.
    '''

    pass