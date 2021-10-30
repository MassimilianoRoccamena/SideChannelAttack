import importlib

ROOT_PACKAGE_NAME = 'main.core'

def package_name(nodes):
    output = nodes[0]
    for node in nodes[1:]:
        output = f"{output}.{node}"
    return f"{ROOT_PACKAGE_NAME}.{output}"

def module_name(package, module):
    return f"{package}.{module}"

def get_class(package, module, class_name):
    parent_name = module_name(package, module)
    print(parent_name)
    print(class_name)
    module = importlib.import_module(parent_name)
    print(dir(module))
    return getattr(module, class_name)