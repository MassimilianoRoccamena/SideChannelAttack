from sca.file.loader import OurTraceLoader
from sca.file.convention2.path import FileIdentifier, file_path

class TraceLoader2(OurTraceLoader):
    '''
    Loader of our power traces from raw file using filesystem convention 2.
    '''

    def build_file_id(self, *args):
        return FileIdentifier(*args)

    def build_file_path(self, file_id):
        return file_path(file_id)