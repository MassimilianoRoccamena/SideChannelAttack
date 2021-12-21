from aidenv.app.dprocess.config import build_task_object1
from aidenv.app.dprocess.config import build_task_object2

def build_task_kwarg(kwarg_name):
    '''
    Decorator for auto-build of a task object kwarg
    kwarg_name: parameter name
    '''
    def build_kwarg(cls, config, prompt, param):
        obj = build_task_object1(config[param], prompt)
        kwargs = {param:obj}
        config = cls.update_kwargs(config, **kwargs)
        return config

    def wrapper1(f):
        def wrapper2(cls, config, prompt):
            return build_kwarg(cls, config, prompt, kwarg_name)

        return wrapper2
    
    return wrapper1