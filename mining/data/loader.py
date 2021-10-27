from core.data.identifier import FileIdentifier
from core.data.path import file_path
from core.data.loader import BasicFileLoader

FILE_ID = FileIdentifier('1.00', '52.000', '32')
FPATH = file_path(FILE_ID)

loader = BasicFileLoader(FPATH)

def print_load_all():
    traces, texts = loader.load_all()
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def print_load_some(trace_idx):
    traces, texts = loader.load_some(trace_idx)
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def print_load_some_projected(trace_idx, sample_idx):
    traces, texts = loader.load_some_projected(trace_idx, sample_idx)
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def print_load():
    print_load_all()
    print_load_some([0, 3, 99, 333, 499])
    print_load_some_projected([40, 400], [i for i in range(100, 10100)])