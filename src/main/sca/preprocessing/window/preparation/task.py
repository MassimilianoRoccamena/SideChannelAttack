import os
from math import ceil
import numpy as np
from tqdm.auto import trange

from utils.persistence import save_json
from aidenv.api.dprocess.config import build_task_kwarg
from aidenv.api.dprocess.task import DataProcess
from sca.file.params import str_hex_bytes
from sca.preprocessing.window.loader import WindowLoader1 as FileConvention1
from sca.preprocessing.window.loader import WindowLoader2 as FileConvention2
from sca.preprocessing.window.slicer import StridedSlicer as Strided
from sca.preprocessing.window.slicer import RandomSlicer as Random
from sca.preprocessing.window.reader import WindowReader

class BasicPreparation(DataProcess):
    '''
    Trace windows data preparation task.
    '''

    INVALID_SUBSET_MSG = 'invalid dataset size type'
    DATA_HEADER = 'voltage,frequency,key_value,plain_index,plain_text,window_start,window_end'

    def __init__(self, loader, voltages, frequencies, key_values,
                    num_plain_texts, size, partitioning, full_info=False, log_dir=None):
        '''
        Create new windows preparation task.
        '''
        self.loader = loader
        self.voltages = list(voltages)
        self.frequencies = list(frequencies)
        if key_values is None:
            key_values = str_hex_bytes()
        self.key_values = list(key_values)
        self.num_plain_texts = num_plain_texts
        self.size = size
        self.partitioning = partitioning
        self.full_info = full_info
        self.log_dir = log_dir

    @classmethod
    @build_task_kwarg('loader')
    def build_kwargs(cls, config, prompt):
        pass

    def run(self):
        # init params
        params = {'voltages':self.voltages,'frequencies':self.frequencies,
                    'key_values':self.key_values,'num_plain_texts':self.num_plain_texts,
                    'size':self.size}
        mapping_enabled = self.partitioning.mapping.enabled
        mapping_bucket = self.partitioning.mapping.bucket_size
        mapping_params = {'enabled':mapping_enabled}
        params['mapping'] = mapping_params
        if mapping_enabled:
            mapping_params['bucket_size'] = mapping_bucket
            print('Using hash-map based data persistence')
        datalen_params = {}
        params['datalen'] = datalen_params

        # split on plain_texts
        root_plain_idx = np.arange(0, self.num_plain_texts)
        np.random.shuffle(root_plain_idx)

        # partitions
        partitions = self.partitioning.sets
        for ip,partition in enumerate(partitions):
            partition_name = partition.name
            partition_size = partition.size
            print('')
            print(f'Building {partition_name} partition')

            # split plain texts
            partition_size = int(partition_size * self.num_plain_texts)
            partition_plain_idx = root_plain_idx[:partition_size]
            if partition_size < len(root_plain_idx):
                if ip == len(partitions)-1: # last partition
                    partition_plain_idx = np.array(list(partition_plain_idx)+list(root_plain_idx[partition_size:]))
                else:
                    root_plain_idx = root_plain_idx[partition_size:]
            print(f'Set has {len(partition_plain_idx)} plain texts')

            # build and save data
            partition_reader = WindowReader(self.loader, self.voltages, self.frequencies, self.key_values, partition_plain_idx)
            partition_read_idx = np.random.choice(len(partition_reader), int(len(partition_reader)*self.size), replace=False) # subset
            num_samples = len(partition_read_idx)
            datalen_params[partition_name] = num_samples
            print(f'Set has {num_samples} data points')
            print('')

            def save_row(f, reader_idx):
                sample_idx = partition_read_idx[reader_idx]
                voltage, frequency, key_value, plain_index, \
                     plain_text, window_start, window_end = partition_reader[sample_idx]
                curr_line_left = f'{voltage},{frequency},{key_value},{plain_index}'
                curr_line_right = f'{plain_text},{window_start},{window_end}'
                print(f'{curr_line_left},{curr_line_right}', file=f)
            
            # populate dataframe(s)
            header_line = BasicPreparation.DATA_HEADER
            if mapping_enabled:
                data_path = lambda buck: os.path.join(self.log_dir, f'{partition_name}{buck}.csv')
                bucket_idx = 0
                f = open(data_path(bucket_idx), 'w')
                print(header_line, file=f)
                for ir in trange(num_samples):
                    if ir % mapping_bucket == 0 and ir != 0:
                        f.close()
                        bucket_idx += 1
                        f = open(data_path(bucket_idx), 'w')
                        print(header_line, file=f)
                    save_row(f, ir)
                f.close()
            else:
                data_path = os.path.join(self.log_dir, f'{partition_name}.csv')
                with open(data_path, 'w') as f:
                    print(header_line, file=f)
                    for ir in trange(num_samples):
                        save_row(f, ir)

        # save params
        params_path = os.path.join(self.log_dir, 'params.json')
        save_json(params, params_path)
            