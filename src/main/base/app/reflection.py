import importlib

from main.base.app.params import CORE_PACKAGE

def get_package_name(package_nodes):
    root_package = CORE_PACKAGE
    package_name = package_nodes[0]
    for node in package_nodes[1:]:
        package_name = f"{package_name}.{node}"
    return f"{root_package}.{package_name}"

def get_module_name(package_name, module_name):
    return f"{package_name}.{module_name}"

def get_class(package_name, module_name, class_name):
    parent_name = get_module_name(package_name, module_name)

    module_name = importlib.import_module(parent_name)
    if module_name is None:
        raise ValueError(f'module {parent_name} not found')

    return getattr(module_name, class_name)