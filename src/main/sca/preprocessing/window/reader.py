from aidenv.api.reader import FileReader

from sca.file.params import str_hex_bytes

class WindowReader(FileReader):
    '''
    Reader of trace windows from raw file. Reading ordering is driven by
    joint values of voltage, frequency, key_value, plain_text, trace_window.
    '''

    INVALID_INDEX_MSG = 'invalid reader index'

    def __init__(self, loader, voltages, frequencies, key_values, plain_indices):
        '''
        Create new reader of trace windows.
        loader: trace windows loader
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_indices: plain texts indices of the encryption
        '''
        self.loader = loader
        self.file_id = None

        self.voltages = voltages
        self.voltage = None

        self.frequencies = frequencies
        self.frequency = None

        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        self.key_values = key_values
        self.key_value = None
        
        self.num_files = len(voltages) * len(frequencies) * len(key_values)

        self.plain_indices = plain_indices
        self.plain_index = None
        self.plain_text = None

        self.num_windows = len(self.loader.slicer)
        self.window_index = None
        self.window_start = None
        self.window_end = None
        self.trace_window = None

        self.num_samples = self.num_files * len(self.plain_indices) * self.num_windows

    def validate_reader_index(self, reader_index):
        '''
        Check consistency of a reader index.
        reader_index: reader index of a window
        '''
        if reader_index < 0 or reader_index >= len(self):
            raise IndexError(WindowReader.INVALID_INDEX_MSG)

    def translate_reader_index(self, reader_index):
        '''
        Translate a reader index into informations for loading the window from raw file,
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
        self.voltage = self.voltages[volt_idx]

        # frequency
        size = int((group[1]-group[0]) / len(self.frequencies))
        freq_idx, group = subindex_group(reader_index, len(self.frequencies), size, group[0])
        self.frequency = self.frequencies[freq_idx]

        # key value
        size = int((group[1]-group[0]) / len(self.key_values))
        kval_idx, group = subindex_group(reader_index, len(self.key_values), size, group[0])
        self.key_value = self.key_values[kval_idx]

        loader = self.loader
        self.file_id = loader.build_file_id(self.voltage, self.frequency, self.key_value)

        # plain text
        num_plains = len(self.plain_indices)
        size = int((group[1]-group[0]) / num_plains)
        plain_idx, group = subindex_group(reader_index, num_plains, size, group[0])
        self.plain_index = self.plain_indices[plain_idx]

        # window
        size = int((group[1]-group[0]) / self.num_windows)
        window_idx, group = subindex_group(reader_index, self.num_windows, size, group[0])
        self.window_index = window_idx

    def format_plain_text(self, plain_text):
        '''
        Format plain text into a hexadecimal string.
        '''
        self.plain_text = ''
        for plain_byte in plain_text:
            plain_byte = '{:02x}'.format(plain_byte)
            self.plain_text = f'{self.plain_text}{plain_byte}'

    def read_sample(self, reader_index):
        '''
        Read a window of a trace using the file loader.
        reader_index: reader index of a window
        '''
        self.translate_reader_index(reader_index)
        loader = self.loader
        loader.set_file_id(self.file_id)
        trace_window, plain_text = loader.load_window_of_some_traces([self.plain_index], self.window_index)
        self.window_start = loader.window_start
        self.window_end = loader.window_end
        self.trace_window = trace_window
        self.format_plain_text(plain_text)
        return trace_window

    def __len__(self):
        return self.num_samples