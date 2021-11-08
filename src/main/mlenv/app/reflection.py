import importlib

from main.utils.reflection import get_package_name
from main.utils.reflection import get_module_path
from main.utils.reflection import get_class
from main.mlenv.app.params import get_core_package

def get_core_package_name(package_nodes):
    root_package = get_core_package()
    return get_package_name(package_nodes, root_package)

def get_core_class(package_name, module_name, class_name):
    module_path = get_module_path(package_name, module_name)
    return get_class(module_path, class_name)