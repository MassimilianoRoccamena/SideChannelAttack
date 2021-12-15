import numpy as np

from aidenv.api.dlearn.config import build_dataset_kwarg
from sca.file.convention1.loader import TraceLoader1
from sca.file.convention2.loader import TraceLoader2

class WindowLoader:
    '''
    Abstract loader of power trace windows from file.
    '''

    def __init__(self, slicer):
        '''
        Create new loader of trace windows.
        slicer: trace slicer
        '''
        super().__init__()
        self.slicer = slicer

    @classmethod
    @build_dataset_kwarg('slicer')
    def build_kwargs(cls, config, prompt):
        pass

    def load_window_of_some_traces(self, trace_indices, window_index):
        '''
        Load a window from some traces of a batch file.
        trace_indices: list of trace indices of a file
        window_index: window index of a trace
        '''
        raise NotImplementedError

class WindowLoader1(WindowLoader, TraceLoader1):
    '''
    Loader of power trace windows using filesystem convention 1.
    '''

    def load_window_of_some_traces(self, trace_indices, window_index):
        start, end = self.slicer[window_index]
        time_idx = np.arange(start, end+1)
        return self.load_some_projected_traces(trace_indices, time_idx)[0][0]

class WindowLoader2(WindowLoader, TraceLoader2):
    '''
    Loader of power trace windows using filesystem convention 2.
    '''

    def load_window_of_some_traces(self, trace_indices, window_index):
        start, end = self.slicer[window_index]
        time_idx = np.arange(start, end+1)
        return self.load_some_projected_traces(trace_indices, time_idx)[0][0]