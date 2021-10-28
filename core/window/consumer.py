import numpy as np

from core.data.identifier import FileIdentifier
from core.data.loader import AdvancedFileLoader

class WindowLoader(AdvancedFileLoader):
    '''
    Loader of trace windows from a batch file given its identifier
    '''

    def __init__(self, slicer):
        '''
        Create new loader of trace windows
        '''
        super().__init__()
        self.slicer = slicer

    def load_some_sliced(self, trace_idx, window_idx):
        '''
        Load a window from a trace of a batch file
        trace_idx: list of trace indices of a file
        window_idx: window index of a trace
        '''
        start, end = self.slicer[window_idx]
        time_idx = np.arange(start, end+1)
        return self.load_some_projected(trace_idx, time_idx)

class WindowConsumer(WindowLoader):
    def __init__(self, slicer, volt, freq, kvalue, ntraces):
        '''
        Create new consumer of trace windows
        '''
        super().__init__(slicer)
        self.volt = volt
        self.freq = freq
        self.kvalue = kvalue
        self.nfiles = len(volt) * len(freq) * len(kvalue)
        self.ntraces = ntraces
        self.file_id = None
        self.trace_idx = None
        self.window_idx = None

    def translate_index(self, consumer_idx):
        '''
        Translate a consumer index into informations for loading from file, ordering the index
        as (voltage, frequency, key value, trace, window)
        consumer_idx: consumer index of a window
        '''
        def subindex_group(idx, count, size, min):
            groups = [[min + x*size, min + (x+1)*size - 1] for x in range(count)]
            for i, group in enumerate(groups):
                if idx < group[1]:
                    return i, group
        
        # voltage
        size = int(len(self) / len(self.volt))
        volt_idx, group = subindex_group(consumer_idx, len(self.volt), size, 0)

        # frequency
        size = int((group[1]-group[0]) / len(self.freq))
        freq_idx, group = subindex_group(consumer_idx, len(self.freq), size, group[0])

        # key value
        size = int((group[1]-group[0]) / len(self.kvalue))
        kvalue_idx, group = subindex_group(consumer_idx, len(self.kvalue), size, group[0])

        self.file_id = FileIdentifier(self.volt[volt_idx], self.freq[freq_idx],
                                        self.kvalue[kvalue_idx])

        # trace
        size = int((group[1]-group[0]) / self.ntraces)
        trace_idx, group = subindex_group(consumer_idx, self.ntraces, size, group[0])
        self.trace_idx = trace_idx

        # window
        size = int((group[1]-group[0]) / len(self.slicer))
        window_idx, group = subindex_group(consumer_idx, len(self.slicer), size, group[0])
        self.window_idx = window_idx

    def consume(self, consumer_idx):
        '''
        Consume a new trace window from a global consumer index
        consumer_idx: consumer index of a window
        '''
        self.translate_index(consumer_idx)
        self.set_file_id(self.file_id)
        return self.load_some_sliced([self.trace_idx], [self.window_idx])

    def __len__(self):
        return self.nfiles * self.ntraces * len(self.slicer)

    def __getitem__(self, idx):
        return self.consume(idx)