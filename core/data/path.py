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

# main batch file of traces naming

def parent_path(volt, freq):
    return os.path.join(root_data_dir,
                        f'{cat_pref_suff(volt,VOLT_SUFFIX)}',
                        f'{cat_pref_suff(freq,FREQ_SUFFIX)}')

def file_name(date, freq, srate, sbits, kid, kvalue, nsamples):
    low = f'{date}_{freq_name(freq)}_{srate_name(srate)}'
    high = f'{sbits_name(sbits)}_{key_name(kid,kvalue)}_{nsamples}'
    return f'{low}_{high}.{BATCH_EXTENSION}'

class FileIdentifier:
    def __init__(self, volt, freq, srate, sbits, kid, kvalue, nsample):
        self.volt = volt
        self.freq = freq
        self.srate = srate
        self.sbits = sbits
        self.kid = kid
        self.kvalue = kvalue
        self.nsample = nsample

def file_path(file_id):
    return os.path.join(parent_path(file_id.volt, file_id.freq),
                        file_name(file_id.date, file_id.freq, file_id.srate, file_id.sbits,
                                    file_id.kid, file_id.kvalue, file_id.nsample))