import pickle
import json
import numpy as np

def load_pickle(path, mode='rb'):
    with open(path, mode) as f:
        return pickle.load(f)

def save_pickle(o, path, mode='wb'):
    with open(path, mode) as f:
        pickle.dump(o, f)

def load_json(path):
    with open(path, 'r') as f:
        s = f.read()
    return json.loads(s)

def save_json(o, path, indent=4):
    s = json.dumps(o, indent=indent)
    with open(path, 'w') as f:
        print(s, file=f)

def load_numpy(path):
    return np.load(path)

def save_numpy(o, path):
    np.save(path, o)