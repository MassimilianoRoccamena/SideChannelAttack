import json

def load_json(path):
    with open(path, 'r') as f:
        s = f.read()
    return json.loads(s)

def save_json(o, path, indent=4):
    s = json.dumps(o, indent=indent)
    with open(path, 'w') as f:
        print(s, file=f)