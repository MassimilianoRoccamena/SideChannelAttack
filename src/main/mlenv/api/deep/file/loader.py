from main.mlenv.api.config import CoreObject

class FileLoader(CoreObject):
    '''
    Abstract core loader of data from file.
    '''

    def set_file_path(self, file_path):
        '''
        Set the data file path.
        file_path: file path
        '''
        raise NotImplementedError

    def set_file_id(self, file_id):
        '''
        Set the data file identifier.
        file_id: file identifier
        '''
        raise NotImplementedError

    def build_file_id(self, *args):
        '''
        Build a file identifier by using some args
        '''
        raise NotImplementedError