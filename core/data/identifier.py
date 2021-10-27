from core.data.params import PathReference

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

class TraceIdentifier:
    def __init__(self, file_id, trace_idx):
        self.file_id = file_id
        self.trace_idx = trace_idx