import os

from core.data.params import PathReference

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

def cat2(s0, s1):
    '''
    Concatenate 2 strings
    '''
    return f'{s0}{s1}'
    
def cat3(s0, s1, s3):
    '''
    Concatenate 3 strings
    '''
    return f'{s0}{s1}{s3}'


def volt_name(value):
    #return cat2(value, VOLT_SUFFIX)        # <-- changed convention
    return cat2(VOLT_PREFIX, value)

def freq_name(value):
    #return cat2(value, FREQ_SUFFIX)        # <-- changed convention
    return cat3(FREQ_PREFIX, value, FREQ_SUFFIX)

def srate_name(value):
    return cat2(value, SRATE_SUFFIX)

def sbits_name(value):
    return cat2(value, SBITS_SUFFIX)

def key_name(bid, value):
    return f"{cat2(KEY_PREFIX, bid)}-{value}"

# traces file path naming

class FileIdentifier:
    '''
    Identification values of a batch file of power traces
    '''
    def __init__(self, volt, freq, kvalue):
        '''
        Create identifier of the file
        volt: voltage used by the device
        freq: frequency used by the device
        kvalue: value of the given key byte
        '''
        self.volt = volt
        self.freq = freq
        self.kvalue = kvalue
        self.date = PathReference.date
        self.mode = PathReference.mode
        self.srate = PathReference.srate
        self.sbits = PathReference.nbits
        self.kid = PathReference.kid
        self.ntraces = PathReference.ntraces

def parent_path(file_id):
    '''
    Build path up to parent directory of the file.
    Used in the past because of previous file path had full path
    function of (volt, freq) instead of only the file name.
    '''
    # <-- changed convention

    #return os.path.join(root_data_dir,
    #                    f'{volt_name(file_id.volt)}',
    #                    f'{freq_name(file_id.freq)}')

    return root_data_dir

def file_name(file_id):
    '''
    Compute name of the file with extension
    '''
    # <-- changed convention

    #s0 = f'{file_id.date}_{freq_name(file_id.freq)}_{srate_name(file_id.srate)}'
    #s1 = f'{sbits_name(file_id.sbits)}_{key_name(file_id.kid,file_id.kvalue)}_{file_id.ntraces}'

    s0 = f'{file_id.date}_{file_id.mode}_{volt_name(file_id.volt)}'
    s1 = f'{freq_name(file_id.freq)}_{srate_name(file_id.srate)}_{sbits_name(file_id.sbits)}'
    s2 = f'{key_name(file_id.kid,file_id.kvalue)}_{file_id.ntraces}'

    return f'{s0}_{s1}_{s2}.{FILE_EXTENSION}'

def file_path(file_id):
    '''
    Compute path of the file
    '''
    return os.path.join(parent_path(file_id), file_name(file_id))