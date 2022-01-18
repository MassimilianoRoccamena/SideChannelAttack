import importlib

MODULE_NOT_FOUND_MSG = lambda mod: f'module {mod} not found'

def get_package_name(package_nodes):
    package_name = None
    for node in package_nodes:
        if package_name is None:
            package_name = node
        else:
            package_name = f"{package_name}.{node}"
    return package_name

def get_module_path(package_name, module_name):
    return f"{package_name}.{module_name}"

def get_class(module_path, class_name):
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(MODULE_NOT_FOUND_MSG(module_path))

    return getattr(module, class_name)