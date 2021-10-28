TRACE_SIZE = 134016
TEXT_SIZE = 16
KEY_SIZE = 16
BYTE_SIZE = 256

class PathReference:
    '''
    Container of data constant parameters used as words
    in the path of a file.
    '''
    date = '2021-10-25'
    mode = '100t_duDFS'       # <-- changed convention
    sampling_rate = '125'
    sampling_bits = '12'
    key_id = '0'
    num_traces = '1k'