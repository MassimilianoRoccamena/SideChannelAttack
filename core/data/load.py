import numpy as np

from core.data.path import file_path

HEAD_SIZE=26

NO_FILE_MSG = "no file path has been specified"

class BasicLoader:
    '''
    Basic loader of power measurements from a batch file of same key encryption traces
    '''

    def __init__(self, fpath):
        '''
        Build new file loader
        fpath: file path
        '''
        self.set_file_path(fpath)

    def set_file_path(self, fpath):
        '''
        Set the binary .dat file path of a batch of key encrypted traces
        fpath: file path
        '''
        self.file_path = fpath

        if self.file_path is None:
            self.ntraces = None
            self.nsamples = None
            self.sample_type
            self.text_len = None
            self.sample_dtype = None
            self.row_len = None
            self.key = None
            return

        with open(self.file_path,'rb') as infile:
            self.ntraces = int.from_bytes(infile.read(4), byteorder='little', signed=False)
            self.nsamples = int.from_bytes(infile.read(4), byteorder='little', signed=False)
            self.sample_type = infile.read(1).decode("ascii")
            self.text_len = int.from_bytes(infile.read(1), byteorder='little', signed=False)
            
            if (self.sample_type=='f'):
                self.sample_dtype=np.dtype('float32')
            elif (self.sample_type=='d'):
                self.sample_dtype=np.dtype('float64')
            else:
                assert(False)
            
            self.row_len = self.text_len + self.nsamples*self.sample_dtype.itemsize
            self.key = np.frombuffer(buffer=infile.read(16), dtype='uint8');
    
    def load_all(self):
        '''
        Get the whole full traces and plain texts from the batch file
        '''
        if self.file_path is None:
            raise ValueError(NO_FILE_MSG)

        with open(self.file_path,'rb') as infile:
            infile.seek(HEAD_SIZE, 0)
            
            texts = np.zeros((self.ntraces, self.text_len), dtype= 'uint8');
            traces = np.zeros((self.ntraces, self.nsamples), dtype= self.sample_dtype);
            
            for i in np.arange(0, self.ntraces):
                traces[i,:] = np.frombuffer(buffer=infile.read(self.nsamples* self.sample_dtype.itemsize), dtype= self.sample_dtype)
                texts[i,:] = np.frombuffer(buffer=infile.read(self.text_len* texts.itemsize), dtype=texts.dtype)
            
        return traces, texts

    def load_some(self, trace_idx):
        '''
        Get some full traces and plain texts from the batch file
        trace_idx: list of traces indices of the file
        '''
        if self.file_path is None:
            raise ValueError(NO_FILE_MSG)

        idx = np.array(trace_idx)
        assert np.all(idx < self.ntraces)
        assert np.all(idx.dtype == 'int')
        n = len(idx)

        with open(self.file_path,'rb') as infile:
            texts = np.zeros((n, self.text_len), dtype= 'uint8');
            traces = np.zeros((n, self.nsamples), dtype= self.sample_dtype);
            j = 0

            for i in idx:
                infile.seek(HEAD_SIZE+ self.row_len*i, 0)
                traces[j,:] = np.frombuffer(buffer=infile.read(self.nsamples* self.sample_dtype.itemsize), dtype= self.sample_dtype)
                texts[j,:] = np.frombuffer(buffer=infile.read(self.text_len* texts.itemsize), dtype=texts.dtype)
                j = j + 1

        return traces, texts
    
    def load_some_projected(self, trace_idx, time_idx):
        '''
        Load some temporal projected traces from the batch file
        trace_idx: list of traces indices of the batch
        time_idx: list of temporal indices of the traces
        '''
        if self.file_path is None:
            raise ValueError(NO_FILE_MSG)

        tr_idx = np.array(trace_idx).reshape((len(trace_idx),1))
        tm_idx = np.array(time_idx)
        
        trace_len = len(trace_idx)
        time_len = len(time_idx)
        
        with open(self.file_path,'rb') as infile:
            traces = np.zeros((trace_len, time_len), dtype= self.sample_dtype)
            texts = np.zeros((trace_len, self.text_len), dtype= 'uint8')
            
            pos = HEAD_SIZE + tr_idx*self.row_len + tm_idx*self.sample_dtype.itemsize
            
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    infile.seek(pos[i][j], 0)
                    traces[i,j] =  np.frombuffer(buffer=infile.read(self.sample_dtype.itemsize), dtype= self.sample_dtype)
                
                infile.seek(HEAD_SIZE+ self.row_len* trace_idx[i] + self.nsamples* self.sample_dtype.itemsize , 0)
                texts[i,:] = np.frombuffer(buffer=infile.read(self.text_len* texts.itemsize), dtype=texts.dtype)
        
        return traces, texts

class AdvancedLoader(BasicLoader):
    '''
    Advanced loader of power measurements from a batch file given its identifier
    '''

    def __init__(self, file_id=None):
        self.set_file_id(file_id)

    def set_file_id(self, file_id):
        self.file_id = file_id
        self.set_file_path(file_path(file_id))