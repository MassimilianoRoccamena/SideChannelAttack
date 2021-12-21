from sca.file.loader import OurTraceLoader
from sca.file.convention1.path import FileIdentifier
from sca.file.convention1.path import file_path

class TraceLoader1(OurTraceLoader):
    '''
    Loader of our power traces from raw file using filesystem convention 1.
    '''

    def __init__(self, file_id=None):
        '''
        Create new loader of traces using convention 1.
        file_id: file identifier
        '''
        self.set_file_id(file_id)

    def set_file_id(self, file_id):
        self.file_id = file_id
        if file_id is None:
            self.set_file_path(None)
        else:
            self.set_file_path(file_path(file_id))

    def build_file_id(self, *args):
        return FileIdentifier(*args)