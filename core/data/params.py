TRACE_SIZE = 134016
TEXT_SIZE = 16
KEY_SIZE = 16
BYTE_SIZE = 256

class PathReference:
    '''
    Container of data constant parameters used as words
    in the path of a file
    '''
    date = '2021-10-25'
    mode = '100t_duDFS'       # <-- changed convention
    srate = '125'
    nbits = '12'
    kid = '0'
    ntraces = '1k'