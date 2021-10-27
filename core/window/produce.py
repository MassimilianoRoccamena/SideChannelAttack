import numpy as np

from core.data.load import AdvancedLoader

class WindowLoader(AdvancedLoader):
    '''
    Loader of traces windows from a batch file given its identifier
    '''

    def __init__(self, slicer):
        super().__init__()
        self.slicer = slicer

    def load_some_sliced(self, trace_idx, window_idx):
        start, end = self.slicer[window_idx]
        time_idx = np.arange(start, end+1)
        return self.load_some_projected(trace_idx, time_idx)

class WindowProducer(WindowLoader):
    def __init__(self, slicer, buffer_size):
        super().__init__(slicer)
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, slicer.window_size))
        self.buffer_idx = 0
        self.cycle_idx = 0
        self.cycle_cnt = 0

    def read(self):
        if self.cycle_idx == 0:
            fetched = self.fetch()
            np.copyto(self.buffer, fetched)
        elif self.cycle_idx == self.slicer.nwindows:
            self.buffer_idx = 0
            self.cycle_idx = 0
            self.cycle_cnt += 1
            self.cycle()
            fetched = self.fetch()
            np.copyto(self.buffer, fetched)
        elif self.buffer_idx == self.buffer_size:
            self.buffer_idx = 0
            fetched = self.fetch()
            np.copyto(self.buffer, fetched)

        val = self.buffer[self.buffer_idx]

        self.buffer_idx += 1
        self.cycle_idx += 1

        return val

    def cycle(self):
        raise NotImplementedError

    def fetch(self):
        raise NotImplementedError

class SameFileProducer(WindowProducer):
    def __init__(self, slicer, buffer_size):
        super().__init__(slicer, buffer_size)

class SameTraceProducer(SameFileProducer):
    def __init__(self, slicer, buffer_size):
        super().__init__(slicer, buffer_size)

class MultiTraceProducer(SameFileProducer):
    def __init__(self, slicer, buffer_size):
        super().__init__(slicer, buffer_size)

class MultiFileProducer(WindowProducer):
    def __init__(self, slicer, buffer_size):
        super().__init__(slicer, buffer_size)