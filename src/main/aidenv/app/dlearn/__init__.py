from aidenv.app.basic.params import set_environment_name
from aidenv.app.basic import run_task
from aidenv.app.basic.config import add_determinism
from aidenv.app.dlearn.config import build_determinism

def run(*args):
    '''
    Entry point for dlearn environment.
    args: program arguments
    '''
    set_environment_name('dlearn')
    add_determinism(build_determinism)
    run_task('Deep learning', *args)