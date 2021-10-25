from core.data.path import file_path
from basic import BasicLoader

class AdvancedLoader(BasicLoader):
    '''
    Advanced loader of power measurements from a batch file of same key encryption traces
    '''

    def __init__(self, file_id):
        self.set_file_id(file_id)

    def set_file_id(self, file_id):
        self.file_id = file_id
        self.set_file_path(file_path(file_id))