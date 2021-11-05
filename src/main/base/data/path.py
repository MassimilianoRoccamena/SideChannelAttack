import os

from main.base.utils.string import cat2, cat3
from main.base.data.params import PathReference

# global vars/constants

VOLT_PREFIX = "VCC-"        # <-- changed convention
#VOLT_SUFFIX = 'V'
FREQ_PREFIX = 'freq-'       # <-- changed convention
FREQ_SUFFIX = 'MHz'
SRATE_SUFFIX = 'MSa'
SBITS_SUFFIX = 'bit'
KEY_PREFIX = 'k'

FILE_EXTENSION = 'dat'

LOCAL_DATA_DIR = '.data'
AIRLAB_DATA_DIR = ''

root_data_dir = LOCAL_DATA_DIR

# basic naming manipulation

def voltage_name(value):
    #return cat2(value, VOLT_SUFFIX)        # <-- changed convention
    return cat2(VOLT_PREFIX, value)

def frequency_name(value):
    #return cat2(value, FREQ_SUFFIX)        # <-- changed convention
    return cat3(FREQ_PREFIX, value, FREQ_SUFFIX)

def sampling_rate_name(value):
    return cat2(value, SRATE_SUFFIX)

def sampling_bits_name(value):
    return cat2(value, SBITS_SUFFIX)

def key_name(key_id, key_value):
    return f"{cat2(KEY_PREFIX, key_id)}-{key_value}"

# traces file path naming

class FileIdentifier:
    '''
    Identification values of a batch file of power traces.
    '''
    def __init__(self, voltage, frequency, key_value):
        '''
        Create identifier of the file.
        voltage: voltage used by the device
        frequency: frequency used by the device
        key_value: value of the given key byte
        '''
        self.voltage = voltage
        self.frequency = frequency
        self.key_value = key_value
        self.date = PathReference.date
        self.mode = PathReference.mode
        self.sampling_rate = PathReference.sampling_rate
        self.sampling_bits = PathReference.sampling_bits
        self.key_id = PathReference.key_id
        self.num_traces = PathReference.num_traces

def parent_path(file_id):
    '''
    Build path up to parent directory of the file.
    Used in the past because of previous file path had full path
    function of (volt, freq) instead of only the file name.
    file_id: identifier of the file
    '''
    # <-- changed convention

    #return os.path.join(root_data_dir,
    #                    f'{volt_name(file_id.volt)}',
    #                    f'{freq_name(file_id.freq)}')

    return root_data_dir

def file_name(file_id):
    '''
    Compute name of the file with extension.
    file_id: identifier of the file
    '''
    # <-- changed convention

    #s0 = f'{file_id.date}_{freq_name(file_id.freq)}_{srate_name(file_id.srate)}'
    #s1 = f'{sbits_name(file_id.sbits)}_{key_name(file_id.kid,file_id.kvalue)}_{file_id.ntraces}'

    s0 = f'{file_id.date}_{file_id.mode}_{voltage_name(file_id.voltage)}_{frequency_name(file_id.frequency)}'
    s1 = f'{sampling_rate_name(file_id.sampling_rate)}_{sampling_bits_name(file_id.sampling_bits)}'
    s2 = f'{key_name(file_id.key_id,file_id.key_value)}_{file_id.num_traces}'

    return f'{s0}_{s1}_{s2}.{FILE_EXTENSION}'

def file_path(file_id):
    '''
    Compute path of the file.
    file_id: identifier of the file
    '''
    return os.path.join(parent_path(file_id), file_name(file_id))