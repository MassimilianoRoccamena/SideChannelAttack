import importlib

from main.base.app.params import ROOT_PACKAGE

def get_package_name(path_nodes):
    output = path_nodes[0]
    for node in path_nodes[1:]:
        output = f"{output}.{node}"
    return f"{ROOT_PACKAGE}.{output}"

def get_module_name(package_name, module_name):
    return f"{package_name}.{module_name}"

def get_class(package_name, module_name, class_name):
    parent_name = get_module_name(package_name, module_name)

    module_name = importlib.import_module(parent_name)
    if module_name is None:
        raise ValueError(f'module {parent_name} not found')

    return getattr(module_name, class_name)

def get_class_from_path(path_nodes, module_name, class_name):
    pass