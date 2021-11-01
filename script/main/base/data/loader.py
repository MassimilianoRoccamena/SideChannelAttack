import numpy as np

from main.base.data.path import file_path

class BasicFileLoader:
    '''
    Loader of power traces from a (same key traces) batch file.
    '''
    
    HEADER_SIZE = 26

    NO_FILE_MSG = 'no file path has been specified'
    INVALID_TRACE_INDICES_MSG = 'invalid trace indices'
    INVALID_TIME_INDICES_MSG = 'invalid temporal indices'

    def __init__(self, file_path):
        '''
        Create new traces batch file loader
        file_path: file path
        '''
        self.set_file_path(file_path)

    def set_file_path(self, file_path):
        '''
        Set the data file path of a batch of key encrypted traces.
        file_path: file path
        '''
        self.file_path = file_path

        if self.file_path is None:
            self.num_traces = None
            self.trace_size = None
            self.channel_type = None
            self.text_len = None
            self.channel_dtype = None
            self.row_len = None
            self.key = None
            return

        with open(self.file_path,'rb') as infile:
            self.num_traces = int.from_bytes(infile.read(4), byteorder='little', signed=False)
            self.trace_size = int.from_bytes(infile.read(4), byteorder='little', signed=False)
            self.channel_type = infile.read(1).decode("ascii")
            self.text_len = int.from_bytes(infile.read(1), byteorder='little', signed=False)
            
            if (self.channel_type=='f'):
                self.channel_dtype=np.dtype('float32')
            elif (self.channel_type=='d'):
                self.channel_dtype=np.dtype('float64')
            else:
                assert(False)
            
            self.row_len = self.text_len + self.trace_size*self.channel_dtype.itemsize
            self.key = np.frombuffer(buffer=infile.read(16), dtype='uint8');

    def validate_trace_indices(self, trace_indices):
        '''
        Check consistency of some trace indices.
        trace_indices: list of trace indices of the file
        '''
        if np.all(trace_indices < 0) or np.all(trace_indices >= self.num_traces):
            raise IndexError(BasicFileLoader.INVALID_TRACE_INDICES_MSG)

    def validate_time_indices(self, time_indices):
        '''
        Check consistency of some time indices.
        time_indices: list of temporal indices of a trace
        '''
        if np.all(time_indices < 0) or np.all(time_indices >= self.trace_size):
            raise IndexError(BasicFileLoader.INVALID_TRACE_INDICES_MSG)

    # -----------------------------------------------------------------------------------------
    
    def load_all_traces(self):
        '''
        Get the whole full traces (and their plain texts) from the batch file.
        '''
        if self.file_path is None:
            raise ValueError(BasicFileLoader.NO_FILE_MSG)

        with open(self.file_path,'rb') as infile:
            infile.seek(BasicFileLoader.HEADER_SIZE, 0)
            
            texts = np.zeros((self.num_traces, self.text_len), dtype= 'uint8');
            traces = np.zeros((self.num_traces, self.trace_size), dtype= self.channel_dtype);
            
            for i in np.arange(0, self.num_traces):
                traces[i,:] = np.frombuffer(buffer=infile.read(self.trace_size* self.channel_dtype.itemsize), dtype= self.channel_dtype)
                texts[i,:] = np.frombuffer(buffer=infile.read(self.text_len* texts.itemsize), dtype=texts.dtype)
            
        return traces, texts

    def load_some_traces(self, trace_indices):
        '''
        Get some full traces (and their plain texts) from the batch file.
        trace_indices: list of trace indices of a file
        '''
        if self.file_path is None:
            raise ValueError(BasicFileLoader.NO_FILE_MSG)

        idx = np.array(trace_indices)
        self.validate_trace_indices(idx)

        n = len(idx)

        with open(self.file_path,'rb') as infile:
            texts = np.zeros((n, self.text_len), dtype= 'uint8');
            traces = np.zeros((n, self.trace_size), dtype= self.channel_dtype);
            j = 0

            for i in idx:
                infile.seek(BasicFileLoader.HEADER_SIZE + self.row_len*i, 0)
                traces[j,:] = np.frombuffer(buffer=infile.read(self.trace_size* self.channel_dtype.itemsize), dtype= self.channel_dtype)
                texts[j,:] = np.frombuffer(buffer=infile.read(self.text_len* texts.itemsize), dtype=texts.dtype)
                j = j + 1

        return traces, texts
    
    def load_some_projected_traces(self, trace_indices, time_indices):
        '''
        Load some temporal projected traces (and their plain texts) from the batch file.
        trace_indices: list of trace indices of a file
        time_indices: list of temporal indices of a trace
        '''
        if self.file_path is None:
            raise ValueError(BasicFileLoader.NO_FILE_MSG)

        trace_idx = np.array(trace_indices).reshape((len(trace_indices),1))
        self.validate_trace_indices(trace_idx)

        time_idx = np.array(time_indices)
        self.validate_time_indices(time_idx)
        
        trace_len = len(trace_indices)
        time_len = len(time_indices)
        
        with open(self.file_path,'rb') as infile:
            traces = np.zeros((trace_len, time_len), dtype= self.channel_dtype)
            texts = np.zeros((trace_len, self.text_len), dtype= 'uint8')
            
            pos = BasicFileLoader.HEADER_SIZE + trace_idx*self.row_len + time_idx*self.channel_dtype.itemsize
            
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    infile.seek(pos[i][j], 0)
                    traces[i,j] =  np.frombuffer(buffer=infile.read(self.channel_dtype.itemsize), dtype= self.channel_dtype)
                
                infile.seek(BasicFileLoader.HEADER_SIZE+ self.row_len* trace_indices[i] + self.trace_size* self.channel_dtype.itemsize , 0)
                texts[i,:] = np.frombuffer(buffer=infile.read(self.text_len* texts.itemsize), dtype=texts.dtype)
        
        return traces, texts

class AdvancedFileLoader(BasicFileLoader):
    '''
    Loader of power traces from a batch file given its identifier.
    '''

    def __init__(self, file_id=None):
        self.set_file_id(file_id)

    def set_file_id(self, file_id):
        '''
        Set the data file identifier, then build the path.
        file_id: file identifier
        '''
        self.file_id = file_id
        if file_id is None:
            self.set_file_path(None)
        else:
            self.set_file_path(file_path(file_id))