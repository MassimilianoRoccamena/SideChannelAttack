import numpy as np

from core.data.path import FileIdentifier

class TraceIdentifier:
    def __init__(self, file_id, trace_idx):
        self.file_id = file_id
        self.trace_idx = trace_idx

class BasicTraceIterator:
    def __init__(self, nfiles, ntraces):
        self.nfiles = nfiles
        self.ntraces = ntraces

    def next_trace(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.nfiles * self.ntraces:
            return self.next_trace()
        else:
            raise StopIteration

class AdvancedTraceIterator(BasicTraceIterator):
    def __init__(self, volt, freq, kvalue, ntraces, shuffle=True):
        super().__init__(len(volt)*len(freq)*len(kvalue), ntraces)
        self.voltages = volt
        self.frequencies = freq
        self.kvalues = kvalue
        self.ntraces = ntraces
        self.shuffle = shuffle
        self.shuffle_indices()

    def shuffle_indices(self):
        '''
        Shuffle indices of the traces
        '''
        if self.shuffle:
            volt_idx = np.random.randint(len(self.voltages), size=self.voltages)
            freq_idx = np.random.randint(len(self.frequencies), size=self.frequencies)
            kval_idx = np.random.randint(len(self.kvalues), size=self.kvalues)
            file_idx = np.array([[v,f,k] for v in volt_idx for f in freq_idx for k in kval_idx])
            np.random.shuffle(file_idx)
            self.files_idx = file_idx
            self.traces_idx = np.random.randint(self.ntraces, size=self.ntraces)
        else: # to remove on implemented
            raise NotImplementedError

    def __iter__(self):
        if self.shuffle:
            self.file_idx = 0
            self.trace_idx = 0
        else:
            raise NotImplementedError
        return self

    def next_trace(self):
        trace_id = None

        if self.shuffle:
            file_idx = self.files_idx[self.file_idx]
            trace_idx = self.traces_idx[self.trace_idx]
            trace_id = TraceIdentifier(FileIdentifier(file_idx[0],file_idx[1],file_idx[2]), trace_idx)

            self.file_idx += 1
        else:
            raise NotImplementedError

        return trace_id