import numpy as np

from core.data.identifier import FileIdentifier
from core.data.loader import AdvancedFileLoader

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

    def load_some_sliced(self, trace_indices, window_index):
        '''
        Load a window from a trace of a batch file.
        trace_indices: list of trace indices of a file
        window_index: window index of a trace
        '''
        start, end = self.slicer[window_index]
        time_idx = np.arange(start, end+1)
        return self.load_some_projected(trace_indices, time_idx)

class WindowReader(WindowLoader):
    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        '''
        Create new reader of trace windows.
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

    def translate_index(self, reader_idx):
        '''
        Translate a reader index into informations for loading the window from a file,
        ordering of the index is (voltage, frequency, key value, trace, window).
        reader_idx: reader index of a window
        '''
        def subindex_group(idx, count, size, min):
            groups = [[min + x*size, min + (x+1)*size - 1] for x in range(count)]
            for i, group in enumerate(groups):
                if idx < group[1]:
                    return i, group
        
        # voltage
        size = int(len(self) / len(self.voltages))
        volt_idx, group = subindex_group(reader_idx, len(self.voltages), size, 0)
        volt = self.voltages[volt_idx]

        # frequency
        size = int((group[1]-group[0]) / len(self.frequencies))
        freq_idx, group = subindex_group(reader_idx, len(self.frequencies), size, group[0])
        freq = self.frequencies[freq_idx]

        # key value
        size = int((group[1]-group[0]) / len(self.key_values))
        kval_idx, group = subindex_group(reader_idx, len(self.key_values), size, group[0])
        kval = self.key_values[kval_idx]

        self.file_id = FileIdentifier(volt, freq, kval)

        # trace
        size = int((group[1]-group[0]) / self.num_traces)
        trace_idx, group = subindex_group(reader_idx, self.num_traces, size, group[0])
        self.trace_index = trace_idx

        # window
        size = int((group[1]-group[0]) / len(self.slicer))
        window_idx, group = subindex_group(reader_idx, len(self.slicer), size, group[0])
        self.window_index = window_idx

    def read_window(self, reader_idx):
        '''
        Read a new trace window from a reader index.
        reader_idx: reader index of a window
        '''
        self.translate_index(reader_idx)
        self.set_file_id(self.file_id)
        return self.load_some_sliced([self.trace_index], [self.window_index])

    def __len__(self):
        return self.num_files * self.num_traces * len(self.slicer)

    def __getitem__(self, idx):
        return self.read_window(idx)