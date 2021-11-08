from main.mlenv.app.deepgym.config import build_dataset_object1
from main.mlenv.app.deepgym.config import build_dataset_object2
from main.mlenv.app.deepgym.config import build_model_object1
from main.mlenv.app.deepgym.config import build_model_object2
from main.mlenv.app.deepgym.config import build_learning_object1
from main.mlenv.app.deepgym.config import build_learning_object2

def build_dataset_kwarg(kwarg_name):
    '''
    Decorator for auto-build of a dataset object kwarg
    kwarg_name: parameter name
    '''
    def build_kwarg(cls, config, prompt, param):
        obj = build_dataset_object1(config[param], prompt)
        kwargs = {param:obj}
        config = cls.update_kwargs(config, **kwargs)
        return config

    def wrapper1(f):
        def wrapper2(cls, config, prompt):
            return build_kwarg(cls, config, prompt, kwarg_name)

        return wrapper2
    
    return wrapper1

def build_model_kwarg(kwarg_name):
    '''
    Decorator for auto-build of a model object kwarg
    kwarg_name: parameter name
    '''
    def build_kwarg(cls, config, prompt, param):
        obj = build_model_object1(config[param], prompt)
        kwargs = {param:obj}
        config = cls.update_kwargs(config, **kwargs)
        return config

    def wrapper1(f):
        def wrapper2(cls, config, prompt):
            return build_kwarg(cls, config, prompt, kwarg_name)

        return wrapper2
    
    return wrapper1