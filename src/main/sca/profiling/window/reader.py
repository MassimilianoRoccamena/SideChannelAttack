from aidenv.api.reader import FileReader

class WindowReader(FileReader):
    '''
    Reader of trace windows from file.
    '''

    INVALID_INDEX_MSG = 'invalid reader index'

    def __init__(self, loader, voltages, frequencies, key_values, trace_indices):
        '''
        Create new reader of trace windows.
        loader: window loader
        voltages: desired voltages
        frequencies: desired frequencies
        key_values: desired key values
        trace_indices: list of trace indices of a file
        '''
        self.loader = loader
        self.voltages = voltages
        self.frequencies = frequencies
        self.key_values = key_values
        self.num_files = len(voltages) * len(frequencies) * len(key_values)
        self.trace_indices = trace_indices
        self.file_id = None
        self.trace_index = None
        self.window_index = None

    def validate_reader_index(self, reader_index):
        '''
        Check consistency of a reader index.
        reader_index: reader index of a window
        '''
        if reader_index < 0 or reader_index >= len(self):
            raise IndexError(WindowReader.INVALID_INDEX_MSG)

    def translate_reader_index(self, reader_index):
        '''
        Translate a reader index into informations for loading the window from the file,
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
        volt = self.voltages[volt_idx]

        # frequency
        size = int((group[1]-group[0]) / len(self.frequencies))
        freq_idx, group = subindex_group(reader_index, len(self.frequencies), size, group[0])
        freq = self.frequencies[freq_idx]

        # key value
        size = int((group[1]-group[0]) / len(self.key_values))
        kval_idx, group = subindex_group(reader_index, len(self.key_values), size, group[0])
        kval = self.key_values[kval_idx]

        loader = self.loader
        self.file_id = loader.build_file_id(volt, freq, kval)

        # trace
        num_traces = len(self.trace_indices)
        size = int((group[1]-group[0]) / num_traces)
        trace_idx, group = subindex_group(reader_index, num_traces, size, group[0])
        self.trace_index = self.trace_indices[trace_idx]

        # window
        slicer = loader.slicer
        size = int((group[1]-group[0]) / len(slicer))
        window_idx, group = subindex_group(reader_index, len(slicer), size, group[0])
        self.window_index = window_idx

    def read_sample(self, reader_index):
        self.translate_reader_index(reader_index)

        loader = self.loader
        loader.set_file_id(self.file_id)
        return loader.load_window_of_some_traces([self.trace_index], self.window_index)

    def __len__(self):
        return self.num_files * len(self.trace_indices) * len(self.loader.slicer)