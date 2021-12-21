from aidenv.api.config import CoreObject

class CoreTask(CoreObject):
    '''
    Abstract core executable task.
    '''

    def run(self, config):
        '''
        Run the object task.
        '''
        raise NotImplementedError