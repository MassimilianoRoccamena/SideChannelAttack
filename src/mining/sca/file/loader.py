from main.sca.file.convention1.path import FileIdentifier, file_path
from main.sca.file.loader import OurTraceLoader

FILE_ID = FileIdentifier('1.00', '52.000', '32')
FILE_PATH = file_path(FILE_ID)

loader = OurTraceLoader(FILE_PATH)

def load_all_shapes():
    traces, texts = loader.load_all_traces()
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def load_some_shapes(trace_idx):
    traces, texts = loader.load_some_traces(trace_idx)
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def load_some_projected_shapes(trace_idx, sample_idx):
    traces, texts = loader.load_some_projected_traces(trace_idx, sample_idx)
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def print_loader_example():
    load_all_shapes()
    load_some_shapes([0, 3, 99, 333, 499])
    load_some_projected_shapes([40, 400], [i for i in range(100, 10100)])