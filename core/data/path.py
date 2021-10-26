import os

# global vars/constants

VOLT_SUFFIX = 'V'
FREQ_SUFFIX = 'MHz'
SRATE_SUFFIX = 'MSa'
SBITS_SUFFIX = 'bit'
KEY_PREFIX = 'k'

BATCH_EXTENSION = 'dat'

LOCAL_DATA_DIR = '.data'
AIRLAB_DATA_DIR = ''

root_data_dir = LOCAL_DATA_DIR

# basic prefix/suffix naming

def cat_pref_suff(pref, suff):
    '''
    Concatenate a prefix and a suffix in a word
    '''
    return f'{pref}{suff}'

def volt_name(value):
    return cat_pref_suff(value, VOLT_SUFFIX)

def freq_name(value):
    return cat_pref_suff(value, FREQ_SUFFIX)

def srate_name(value):
    return cat_pref_suff(value, SRATE_SUFFIX)

def sbits_name(value):
    return cat_pref_suff(value, SBITS_SUFFIX)

def key_name(bid, value):
    return f"{cat_pref_suff(KEY_PREFIX, bid)}-{value}"

# main batch file naming

class FileIdentifier:
    '''
    Identification values of a batch file of power traces
    '''
    def __init__(self, date, volt, freq, srate, sbits, kid, kvalue, ntrace):
        '''
        Create identifier of the file
        date: date of the measurements
        volt: voltage used by the device
        freq: frequency used by the device
        srate: sampling rate used for the measurements
        sbit: bits resolution of the sampling
        kid: key byte leanking in the trace
        kvalue: value of the given key byte
        nsample: number of traces in the batch
        '''
        self.date = date
        self.volt = volt
        self.freq = freq
        self.srate = srate
        self.sbits = sbits
        self.kid = kid
        self.kvalue = kvalue
        self.ntrace = ntrace

def parent_path(file_id):
    '''
    Build path up to parent directory of the file
    '''
    return os.path.join(root_data_dir,
                        f'{cat_pref_suff(file_id.volt,VOLT_SUFFIX)}',
                        f'{cat_pref_suff(file_id.freq,FREQ_SUFFIX)}')

def file_name(file_id):
    '''
    Compute name of the file with extension
    '''
    low = f'{file_id.date}_{freq_name(file_id.freq)}_{srate_name(file_id.srate)}'
    high = f'{sbits_name(file_id.sbits)}_{key_name(file_id.kid,file_id.kvalue)}_{file_id.ntrace}'
    return f'{low}_{high}.{BATCH_EXTENSION}'

def file_path(file_id):
    '''
    Compute path of the file
    '''
    return os.path.join(parent_path(file_id), file_name(file_id))