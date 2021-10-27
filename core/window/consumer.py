from copy import copy
import numpy as np

from core.data.loader import AdvancedFileLoader

class WindowLoader(AdvancedFileLoader):
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

class WindowConsumer(WindowLoader):
    def __init__(self, slicer):
        super().__init__(slicer)
        self.cycle_idx = 0
        self.cycle_cnt = 0
        self.on_cycle_start()

    def on_cycle(self):
        if self.cycle_idx == self.slicer.nwindows:
            self.on_cycle_end()
            self.on_cycle_start()

        self.cycle_idx += 1

    def on_cycle_start(self):
        self.cycle()

    def on_cycle_end(self):
        self.cycle_idx = 0
        self.cycle_cnt += 1

    def cycle(self):
        raise NotImplementedError

    def load_next(self):
        raise NotImplementedError

    def consume(self):
        windows = self.load_next()
        self.on_cycle()
        return windows

class SameFileConsumer(WindowConsumer):
    def __init__(self, slicer):
        super().__init__(slicer)

    def cycle(self):
        pass

    def load_next(self):
        pass

class SameTraceConsumer(SameFileConsumer):
    def __init__(self, slicer):
        super().__init__(slicer)

class MultiTraceConsumer(SameFileConsumer):
    def __init__(self, slicer):
        super().__init__(slicer)

class MultiFileConsumer(WindowConsumer):
    def __init__(self, slicer):
        super().__init__(slicer)