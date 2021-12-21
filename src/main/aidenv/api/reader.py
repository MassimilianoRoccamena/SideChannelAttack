class FileReader:
    '''
    Abstract reader of samples from file.
    '''

    def __init__(self, loader):
        '''
        Create new reader of samples from file
        loader: file loader
        '''
        self.loader = loader

    def read_sample(self, reader_index):
        '''
        Read a dataset sample from a file using the file loader and a reader index, keeping
        track all informations of the last sample read.
        reader_index: reader index of a sample
        '''
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.read_sample(index)