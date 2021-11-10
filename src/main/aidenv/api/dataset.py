from aidenv.api.config import CoreObject

class CoreDataset(CoreObject):
    ''''
    Abstract core dataset.
    '''

    def __init__(self, reader):
        '''
        Create new core dataset.
        reader: file reader
        '''
        self.reader = reader

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        raise NotImplementedError