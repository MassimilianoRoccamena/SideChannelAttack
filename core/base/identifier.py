from core.base.params import PathReference

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