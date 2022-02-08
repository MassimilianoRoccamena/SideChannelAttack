import numpy as np

from sca.loader import *

class WindowLoader:
    '''
    Loader from lookup file of power trace windows.
    '''

    def load_trace_window(self, file_path, plain_index, window_start, window_end):
        '''
        Load a window (and corresponding plain text) of a trace from a lookup file.
        file_path: path of the file
        plain_index: plain text index of a file
        window_start: start time index of a window
        window_end: end time index of a window
        '''
        time_idx = np.arange(window_start, window_end+1)
        traces, plain_texts, key = self.load_some_projected_traces(file_path, [plain_index], time_idx)
        return traces[0], plain_texts[0], key

class WindowLoader1(WindowLoader, FileConvention1):
    '''
    Loader of power trace windows using filesystem convention 1.
    '''

    pass

class WindowLoader2(WindowLoader, FileConvention2):
    '''
    Loader of power trace windows using filesystem convention 2.
    '''

    pass