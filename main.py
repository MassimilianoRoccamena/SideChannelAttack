from core.data.path import file_path
from core.data.load.basic import BasicLoader

FPATH = file_path('2021-09-21', '1_00', '50', '125', '12', '0', '63', '1k')

loader = BasicLoader(FPATH)

def test_load_all():
    traces, texts = loader.load_all()
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def test_load_some(trace_idx):
    traces, texts = loader.load_some(trace_idx)
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

def test_load_some_projected(trace_idx, sample_idx):
    traces, texts = loader.load_some_projected(trace_idx, sample_idx)
    print(f'traces have shape {traces.shape}')
    print(f'texts have shape {texts.shape}')

if __name__ == "__main__":
    test_load_all()
    test_load_some([0, 3, 99, 888, 999])
    test_load_some_projected([50, 500], [i for i in range(100, 10100)])