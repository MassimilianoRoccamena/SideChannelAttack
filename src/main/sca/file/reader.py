import numpy as np

from aidenv.api.reader import FileReader
from sca.file.params import str_hex_bytes
from sca.assembler import DynamicAssembler

class TraceReader(FileReader):
    '''
    Reader of power traces from raw file. Reading ordering is driven by
    joint values of voltage, frequency, key_value, plain_text.
    '''

    def __init__(self, loader, voltages, frequencies, key_values, plain_indices):
        '''
        Create new raw reader of static power traces.
        loader: power trace loader
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_indices: plain texts indices of the encryption
        '''
        super().__init__(loader)

        self.voltages = voltages
        self.frequencies = frequencies
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        self.key_values = key_values
        self.plain_indices = plain_indices
        
        self.num_files = len(voltages) * len(frequencies) * len(key_values)
        self.num_samples = self.num_files * len(plain_indices)

    def translate_reader_index(self, reader_index):
        self.validate_reader_index(reader_index)
        
        size = int(len(self) / len(self.voltages))
        volt_idx, group = self.subindex_group(reader_index, len(self.voltages), size, 0)
        voltage = self.voltages[volt_idx]

        size = int((group[1]-group[0]) / len(self.frequencies))
        freq_idx, group = self.subindex_group(reader_index, len(self.frequencies), size, group[0])
        frequency = self.frequencies[freq_idx]

        size = int((group[1]-group[0]) / len(self.key_values))
        kval_idx, group = self.subindex_group(reader_index, len(self.key_values), size, group[0])
        key_value = self.key_values[kval_idx]

        file_id = self.loader.build_file_id(voltage, frequency, key_value)
        file_path = self.loader.build_file_path(file_id)

        size = int((group[1]-group[0]) / len(self.plain_indices))
        plain_idx, group = self.subindex_group(reader_index, len(self.plain_indices), size, group[0])
        plain_index = self.plain_indices[plain_idx]

        return file_path, voltage, frequency, key_value, plain_index

    def read_sample(self, reader_index):
        file_path, voltage, frequency, \
            key_value, plain_index = self.translate_reader_index(reader_index)

        traces, plain_text, key = self.loader.fetch_traces(file_path, [plain_index])

        return voltage, frequency, key_value, plain_index, traces[0], plain_text, key

    def __len__(self):
        return self.num_samples

class DynamicTraceReader(FileReader):
    '''
    Reader of dynamic power traces from raw file. Reading ordering is driven by
    joint values of key_value, plain_text. Reader simulates DFS mechanism by
    switching frequencies with a normal distribution, thi is done by assembling
    windows of static frequency traces.
    '''

    def __init__(self, loader, voltage, frequencies, key_values, plain_indices,
                        num_dynamics, mu, sigma, min_window_len=None, max_window_len=None):
        '''
        Create new raw reader of dynamic power traces.
        loader: power trace loader
        voltage: device voltage
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_indices: plain texts indices of the encryption
        num_dynamics: number of dynamic traces for some given key_value, plain_text
        mu: mean of the gaussian duration of the static windows
        sigma: std of the gaussian duration of the static windows
        min_window_len: min static window size
        max_window_len: max static window size
        '''
        super().__init__(loader)

        self.voltage = voltage
        self.frequencies = frequencies
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        self.key_values = key_values
        self.plain_indices = plain_indices
        self.num_dynamics = num_dynamics
        self.assembler = DynamicAssembler(loader, self.plain_indices, voltage, frequencies, \
                                        mu, sigma, min_window_len, max_window_len)

        self.num_samples = len(key_values) * len(plain_indices) * num_dynamics

    def translate_reader_index(self, reader_index):
        self.validate_reader_index(reader_index)

        size = int(len(self) / len(self.key_values))
        kval_idx, group = self.subindex_group(reader_index, len(self.key_values), size, 0)
        key_value = self.key_values[kval_idx]

        file_id = self.loader.build_file_id(voltage, frequency, key_value)
        file_path = self.loader.build_file_path(file_id)

        size = int((group[1]-group[0]) / len(self.plain_indices))
        plain_idx, group = self.subindex_group(reader_index, len(self.plain_indices), size, group[0])
        plain_index = self.plain_indices[plain_idx]

        return key_value, plain_index

    def read_sample(self, reader_index):
        reader_index = reader_index // self.num_dynamics
        key_value, plain_index = self.translate_reader_index(reader_index)
        traces = self.assembler.make_traces(key_value)

        return key_value, plain_index, traces[0]

    def __len__(self):
        return self.num_samples