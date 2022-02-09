import os
import numpy as np
import pandas as pd

from aidenv.api.config import get_program_log_dir
from sca.file.params import TRACE_SIZE

class TraceAssembler:
    '''
    Object which encapsulate a trace creation strategy.
    '''

    def __init__(self, loader, plain_indices, trace_len):
        self.loader = loader
        self.plain_indices = plain_indices
        self.num_plain_texts = plain_indices[-1] - plain_indices[0] + 1
        if trace_len is None:
            self.trace_len = TRACE_SIZE
        else:
            self.trace_len = trace_len

    def make_traces(self, *args):
        raise NotImplementedError

class StaticAssembler(TraceAssembler):
    '''
    '''

    def make_traces(self, *args):
        voltage = args[0]
        frequency = args[1]
        key_value = args[2]

        file_id = self.loader.build_file_id(voltage, frequency, key_value)
        file_path = self.loader.build_file_path(file_id)

        if self.trace_len is None:
            traces, plain_texts, key = self.loader.load_some_traces(file_path, self.plain_indices)
        else:
            time_idx = np.arange(0, self.trace_len)
            traces, plain_text, key = self.loader.load_some_projected_traces(file_path, self.plain_indices, time_idx)

        return traces

class DynamicAssembler(TraceAssembler):
    '''
    '''

    def __init__(self, loader, plain_indices, voltage, frequencies, mu, sigma,
                    trace_len, min_window_len, max_window_len):
        super().__init__(loader, plain_indices, trace_len)
        self.voltage = voltage
        self.frequencies = frequencies
        self.mu = mu
        self.sigma = sigma
        if min_window_len is None:
            self.min_window_len = 10
        else:
            self.min_window_len = min_window_len
        if max_window_len is None:
            self.max_window_len = 50000
        else:
            self.max_window_len = max_window_len
        self.log_dir = os.path.join(get_program_log_dir(), 'assembler')
        os.mkdir(self.log_dir)

    def on_key_windows(self, key_value, df_windows):
        file_path = os.path.join(self.log_dir, f'{key_value}.csv')
        df_windows.to_csv(file_path, index=False)
    
    def make_traces(self, *args):
        key_value = args[0]

        traces = np.ones((self.num_plain_texts, self.trace_len), dtype='float32')
        all_freq_index = np.arange(len(self.frequencies))
        df_windows = pd.DataFrame(columns=['plain_index','time_start','time_end','frequency'])

        for plain_idx in self.plain_indices:
            plain_idx -= self.plain_indices[0]
            curr_switches = []
            curr_start = 0

            while True:
                window_len = int(self.sigma * np.random.randn() + self.mu)
                window_len = max(window_len, self.min_window_len)
                window_len = min(window_len, self.max_window_len)

                if curr_start + window_len > self.trace_len:
                    curr_switches.append(self.trace_len)
                    break
                else:
                    curr_start += window_len
                    curr_switches.append(curr_start)

            curr_freq = int(np.random.choice(all_freq_index, 1)[0])
            other_freqs = np.delete(all_freq_index, curr_freq)
            time_start = 0

            for switch in curr_switches:
                # current window
                file_id = self.loader.build_file_id(self.voltage, self.frequencies[curr_freq], key_value)
                file_path = self.loader.build_file_path(file_id)
                time_idx = np.arange(time_start, switch)
                windows, _, _ = self.loader.load_some_projected_traces(file_path, [plain_idx], time_idx)
                traces[plain_idx, time_start:switch] = windows[0]
                df_windows = df_windows.append({'plain_index':plain_idx,'time_start':time_start, \
                                    'time_end':switch-1, 'frequency':self.frequencies[curr_freq]}, \
                                    ignore_index=True)

                # switch next
                curr_freq = int(np.random.choice(other_freqs, 1)[0])
                other_freqs = np.delete(all_freq_index, curr_freq)
                time_start = switch

        self.on_key_windows(key_value, df_windows)
        return traces