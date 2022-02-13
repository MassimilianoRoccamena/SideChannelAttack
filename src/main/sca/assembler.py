import numpy as np
import pandas as pd

from sca.file.params import TRACE_SIZE

class TraceAssembler:
    '''
    Object which encapsulate a trace creation strategy.
    '''

    def __init__(self, loader, plain_indices):
        self.loader = loader
        self.plain_indices = plain_indices
        self.num_plain_texts = plain_indices[-1] - plain_indices[0] + 1

    def make_traces(self, *args):
        '''
        Load and process some traces.
        '''
        raise NotImplementedError

class StaticAssembler(TraceAssembler):
    '''
    Static trace assembler, basically a wrapper for a trace loader.
    '''

    def make_traces(self, *args):
        voltage = args[0]
        frequency = args[1]
        key_value = args[2]

        file_id = self.loader.build_file_id(voltage, frequency, key_value)
        file_path = self.loader.build_file_path(file_id)
        traces, plain_text, key = self.loader.fetch_traces(file_path, self.plain_indices)

        return traces

class DynamicAssembler(TraceAssembler):
    '''
    Dynamic trace assembler, creating DFS traces by concatenation of static
    frequency windows.
    '''

    def __init__(self, loader, plain_indices, voltage, frequencies, mu, sigma,
                    min_window_len, max_window_len, track_windows=None):
        '''
        Create new dynamic trace assembler.
        loader: power trace loader
        voltage: device voltage
        frequencies: frequencies switched in the traces
        mu: mean of the gaussian duration of the static windows
        sigma: std of the gaussian duration of the static windows
        min_window_len: min static window size
        max_window_len: max static window size
        track_windows: wheter to track static windows in a dataframe
        '''
        super().__init__(loader, plain_indices)
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
        if track_windows is None:
            self.track_windows = False
        else:
            self.track_windows = True
    
    def make_traces(self, *args):
        key_value = args[0]
        trace_len = self.loader.trace_len

        traces = np.ones((self.num_plain_texts, trace_len), dtype='float32')
        all_freq_index = np.arange(len(self.frequencies))
        if self.track_windows:
            df_windows = pd.DataFrame(columns=['plain_index','time_start','time_end','frequency'])

        for plain_idx in self.plain_indices:
            plain_idx -= self.plain_indices[0]
            curr_switches = []
            curr_start = 0

            while True:
                window_len = int(self.sigma * np.random.randn() + self.mu)
                window_len = max(window_len, self.min_window_len)
                window_len = min(window_len, self.max_window_len)

                if curr_start + window_len > trace_len:
                    curr_switches.append(trace_len)
                    break
                else:
                    curr_start += window_len
                    curr_switches.append(curr_start)

            curr_freq = int(np.random.choice(all_freq_index, 1)[0])
            other_freqs = np.delete(all_freq_index, curr_freq)
            prev_switch = 0
            time_elapsed = 0.
            time_start = 0

            for switch in curr_switches:
                # current window
                delta_t = 1. / float(self.frequencies[curr_freq])
                time_start = int(time_elapsed / delta_t)
                time_end = time_start + switch-prev_switch
                file_id = self.loader.build_file_id(self.voltage, self.frequencies[curr_freq], key_value)
                file_path = self.loader.build_file_path(file_id)
                if time_end > TRACE_SIZE:
                    raise RuntimeError('encountered time end index greater than trace length')
                time_idx = np.arange(time_start, time_end)
                windows, _, _ = self.loader.load_some_projected_traces(file_path, [plain_idx], time_idx)
                traces[plain_idx, prev_switch:switch] = windows[0]

                if self.track_windows:
                    df_windows = df_windows.append({'plain_index':plain_idx,'time_start':prev_switch, \
                                        'time_end':switch, 'frequency':self.frequencies[curr_freq]}, \
                                        ignore_index=True)

                # switch next
                curr_freq = int(np.random.choice(other_freqs, 1)[0])
                other_freqs = np.delete(all_freq_index, curr_freq)
                prev_switch = switch
                time_elapsed += delta_t * (switch-time_start)

        if self.track_windows:
            self.df_windows = df_windows
        return traces

class DynamicAssemblerLoader:
    '''
    Assembled dynamic traces loader.
    '''

    def __init__(self, loader, voltage, assembler_path):
        '''
        Create new assembled dynamic trace loader.
        loader: power trace loader
        voltage: device voltage
        assembler_path: assembled traces lookup data
        '''
        self.loader = loader
        self.voltage = voltage
        self.assembler_path = assembler_path
        self.df_key = None
        self.curr_key = None
    
    def fetch_trace(self, key_value, plain_index):
        trace_len = self.loader.trace_len
        if self.curr_key != key_value:
            self.df_key = pd.read_csv(f'{key_value}.csv')
            self.curr_key = key_value

        df_plain = self.df_key[self.df_key['plain_index']==plain_index]
        time_indices = df_plain[['time_start','time_end']].to_numpy()
        frequencies = df_plain['frequency'].to_numpy()
        n_switches = frequencies.shape[0]

        trace = np.ones(trace_len, dtype='float32')
        prev_end = 0
        time_elapsed = 0.
        plain_text = None

        for i in range(n_switches):
            curr_idx = time_indices[:,i]
            curr_start = curr_idx[0]
            curr_end = curr_idx[1]
            curr_freq = frequencies[i]

            # current window
            delta_t = 1. / float(curr_freq)
            time_start = int(time_elapsed / delta_t)
            time_end = time_start + curr_end-prev_end
            file_id = self.loader.build_file_id(self.voltage, curr_freq, key_value)
            file_path = self.loader.build_file_path(file_id)
            if time_end > TRACE_SIZE:
                raise RuntimeError('encountered time end index greater than trace length')
            time_idx = np.arange(time_start, time_end)
            windows, plain_texts, key = self.loader.load_some_projected_traces(file_path, plain_index, time_idx)
            if plain_text is None:
                plain_text = plain_texts[0]

            trace[plain_index, curr_start:curr_end] = windows[0]

            # switch next
            prev_end = curr_end
            time_elapsed += delta_t * (curr_end-time_start)

        return trace, plain_text, key