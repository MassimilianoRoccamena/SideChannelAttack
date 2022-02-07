import os
import pandas as pd

from utils.persistence import load_json
from aidenv.api.reader import FileReader

class WindowReader(FileReader):
    '''
    Reader of trace windows from data lookup file(s). Reading ordering is driven
    by rows ordering in the lookup file(s).
    '''

    USED_COLS = ['voltage','frequency','key_value','plain_index','window_start','window_end']
    USED_COLS_TYPE = {USED_COLS[0]:'string', USED_COLS[1]:'string',
                        USED_COLS[2]:'string', USED_COLS[3]:'int',
                        USED_COLS[4]:'int', USED_COLS[5]:'int'}

    def __init__(self, loader, data_path, set_name):
        '''
        Create new lookup reader of trace windows.
        loader: trace windows loader
        data_path: path of data
        set_name:: name of one data partition
        '''
        super().__init__(loader)
        self.data_path = data_path
        self.set_name = set_name

        params_path = os.path.join(data_path, 'params.json')
        params = load_json(params_path)
        if params['mapping']['enabled']:
            self.mapping_enabled = True
            self.mapping_bucket = params['mapping']['bucket_size']
        else:
            self.mapping_enabled = False

        self.num_samples = params['datalen'][set_name]
        self.voltages = params['voltages']
        self.frequencies = params['frequencies']
        self.key_values = params['key_values']

    def translate_reader_index(self, reader_index):
        self.validate_reader_index(reader_index)

        used_cols = WindowReader.USED_COLS
        cols_types = WindowReader.USED_COLS_TYPE
        
        if self.mapping_enabled:
            bucket_idx = int(reader_index/self.mapping_bucket)
            df_path = os.path.join(self.data_path, f'{self.set_name}{bucket_idx}.csv')
            skiprows = reader_index - bucket_idx*self.mapping_bucket
        else:
            df_path = os.path.join(self.data_path, f'{self.set_name}.csv')
            skiprows = reader_index

        data = pd.read_csv(df_path, dtype=cols_types, usecols=used_cols,
                            skiprows=range(1,1+skiprows), nrows=1)
        data = data.iloc[[0]]
        voltage = data[used_cols[0]][0]
        frequency = data[used_cols[1]][0]
        key_value = data[used_cols[2]][0]
        plain_index = data[used_cols[3]][0]
        window_start = data[used_cols[4]][0]
        window_end = data[used_cols[5]][0]

        file_id = self.loader.build_file_id(voltage, frequency, key_value)
        file_path = self.loader.build_file_path(file_id)

        return file_path, voltage, frequency, key_value, plain_index, window_start, window_end

    def read_sample(self, reader_index):
        file_path, voltage, frequency, key_value, \
            plain_index, window_start, window_end = self.translate_reader_index(reader_index)
        trace_window, plain_text, key = self.loader.load_trace_window(file_path, plain_index,
                                                                    window_start, window_end)
        return voltage, frequency, key_value, plain_text, trace_window

    def __len__(self):
        return self.num_samples