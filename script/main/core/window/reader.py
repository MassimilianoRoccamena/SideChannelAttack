import numpy as np

from main.base.data.path import FileIdentifier
from main.base.data.loader import AdvancedFileLoader

class WindowLoader(AdvancedFileLoader):
    '''
    Loader of trace windows from a batch file given its identifier.
    '''

    def __init__(self, slicer):
        '''
        Create new loader of trace windows.
        slicer: window slicing strategy
        '''
        super().__init__()
        self.slicer = slicer

    def load_window_of_some_traces(self, trace_indices, window_index):
        '''
        Load a window from some traces of a batch file.
        trace_indices: list of trace indices of a file
        window_index: window index of a trace
        '''
        start, end = self.slicer[window_index]
        time_idx = np.arange(start, end+1)
        return self.load_some_projected_traces(trace_indices, time_idx)[0][0]

class WindowReader(WindowLoader):
    '''
    Reader of trace windows with an indexing strategy which makes easier loading data from file
    and tracking target values in memory.
    '''
    INVALID_INDEX_MSG = 'invalid reader index'

    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        '''
        Create new reader of trace windows.
        slicer: window slicing strategy
        voltages: desired voltages
        frequencies: desired frequencies
        key_values: desired key values
        num_traces: number of traces in each file
        '''
        super().__init__(slicer)
        self.voltages = voltages
        self.frequencies = frequencies
        self.key_values = key_values
        self.num_files = len(voltages) * len(frequencies) * len(key_values)
        self.num_traces = num_traces
        self.file_id = None
        self.trace_index = None
        self.window_index = None

    def validate_reader_index(self, reader_index):
        '''
        Check consistency of a reader index.
        reader_index: reader index of a window
        '''
        if reader_index < 0 or reader_index >= len(self):
            raise IndexError(WindowReader.INVALID_INDEX_MSG)

    def translate_reader_index(self, reader_index):
        '''
        Translate a reader index into informations for loading the window from the file,
        index ordering is (voltage, frequency, key value, trace, window).
        reader_index: reader index of a window
        '''
        self.validate_reader_index(reader_index)

        def subindex_group(idx, count, size, min):
            groups = [[min + x*size, min + (x+1)*size] for x in range(count)]
            for i, group in enumerate(groups):
                if idx < group[1]:
                    return i, group
        
        # voltage
        size = int(len(self) / len(self.voltages))
        volt_idx, group = subindex_group(reader_index, len(self.voltages), size, 0)
        volt = self.voltages[volt_idx]

        # frequency
        size = int((group[1]-group[0]) / len(self.frequencies))
        freq_idx, group = subindex_group(reader_index, len(self.frequencies), size, group[0])
        freq = self.frequencies[freq_idx]

        # key value
        size = int((group[1]-group[0]) / len(self.key_values))
        kval_idx, group = subindex_group(reader_index, len(self.key_values), size, group[0])
        kval = self.key_values[kval_idx]

        self.file_id = FileIdentifier(volt, freq, kval)

        # trace
        size = int((group[1]-group[0]) / self.num_traces)
        trace_idx, group = subindex_group(reader_index, self.num_traces, size, group[0])
        self.trace_index = trace_idx

        # window
        size = int((group[1]-group[0]) / len(self.slicer))
        window_idx, group = subindex_group(reader_index, len(self.slicer), size, group[0])
        self.window_index = window_idx

    def read_window(self, reader_index):
        '''
        Read a new trace window from a reader index.
        reader_index: reader index of a window
        '''
        self.translate_reader_index(reader_index)
        self.set_file_id(self.file_id)
        return self.load_window_of_some_traces([self.trace_index], self.window_index)

    def __len__(self):
        return self.num_files * self.num_traces * len(self.slicer)

    def __getitem__(self, index):
        return self.read_window(index)