class FileReader:
    '''
    Abstract reader of samples from file.
    '''

    INVALID_INDEX_MSG = 'invalid reader index'

    def __init__(self, loader):
        '''
        Create new reader of samples from file
        loader: file loader
        '''
        self.loader = loader

    def validate_reader_index(self, reader_index):
        '''
        Check consistency of a reader index.
        reader_index: reader index of a window
        '''
        if reader_index < 0 or reader_index >= len(self):
            raise IndexError(FileReader.INVALID_INDEX_MSG)

    def subindex_group(self, idx, count, size, min):
        '''
        Subroutine used for reader index translation.
        '''
        groups = [[min + x*size, min + (x+1)*size] for x in range(count)]
        for i, group in enumerate(groups):
            if idx < group[1]:
                return i, group

    def translate_reader_index(self, reader_index):
        '''
        Translate a reader index into informations for loading data from raw file.
        reader_index: reader index of a window
        '''
        raise NotImplementedError

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