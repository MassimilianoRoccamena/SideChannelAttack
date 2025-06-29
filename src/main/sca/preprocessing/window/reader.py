from aidenv.api.reader import FileReader

from sca.file.params import str_hex_bytes

class WindowReader(FileReader):
    '''
    Reader of trace windows from raw file. Reading ordering is driven by
    joint values of voltage, frequency, key_value, plain_text, trace_window.
    '''

    def __init__(self, loader, voltages, frequencies, key_values, plain_indices):
        '''
        Create new raw reader of trace windows.
        loader: trace windows loader
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_indices: plain texts indices of the encryption
        '''
        super().__init__(loader)
        self.num_files = len(voltages) * len(frequencies) * len(key_values)

        self.voltages = voltages
        self.frequencies = frequencies
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        self.key_values = key_values
        self.plain_indices = plain_indices
        self.num_windows = len(self.loader.slicer)

        self.num_samples = self.num_files * len(plain_indices) * self.num_windows

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

        num_plains = len(self.plain_indices)
        size = int((group[1]-group[0]) / num_plains)
        plain_idx, group = self.subindex_group(reader_index, num_plains, size, group[0])
        plain_index = self.plain_indices[plain_idx]

        size = int((group[1]-group[0]) / self.num_windows)
        window_idx, group = self.subindex_group(reader_index, self.num_windows, size, group[0])
        window_index = window_idx

        return file_path, voltage, frequency, key_value, plain_index, window_index

    def format_plain_text(self, plain_text):
        '''
        Format plain text into a hexadecimal string.
        plain_text: binary values of the plain text.
        '''
        out = ''
        for plain_byte in plain_text:
            plain_byte = '{:02x}'.format(plain_byte)
            out = f'{out}{plain_byte}'
        return out

    def read_sample(self, reader_index):
        file_path, voltage, frequency, key_value, \
            plain_index, window_index = self.translate_reader_index(reader_index)
        window_start, window_end, \
            trace_window, plain_text, key = self.loader.load_trace_window(file_path,
                                                                    plain_index, window_index)
        plain_text = self.format_plain_text(plain_text)
        return voltage, frequency, key_value, plain_index, plain_text, window_start, window_end

    def __len__(self):
        return self.num_samples