from aidenv.app.basic.params import set_environment_name
from aidenv.app.basic import run_task

def run(*args):
    '''
    Entry point for dprocess environment.
    args: program arguments
    '''
    set_environment_name('dprocess')
    run_task('Data processing', *args)