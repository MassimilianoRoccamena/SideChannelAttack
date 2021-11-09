from sca.core.window.loader import WindowLoader1
from sca.core.window.slicer import StridedSlicer
from sca.core.window.classification.dataset import MultiClassification

WINDOW_SIZE = 1000
STRIDE = 50

VOLTAGES = ['1.00']
FREQUENCIES = ['52.000']
KEY_VALUES = ['32','a3']
NUM_TRACES = 500

def print_window_classification_item():
    slicer = StridedSlicer(WINDOW_SIZE, STRIDE)
    loader = WindowLoader1(slicer)
    dataset = MultiClassification(loader, VOLTAGES, FREQUENCIES, KEY_VALUES, NUM_TRACES)
    reader = dataset.reader

    reader_idx = 1400000
    window, label = dataset[reader_idx]

    print(f'dataset index is {reader_idx}')
    print(f'window shape is {window.shape}')
    print(f'voltage is {reader.file_id.voltage}')
    print(f'frequency is {reader.file_id.frequency}')
    print(f'key value is {reader.file_id.key_value}')
    print(f'trace index is {reader.trace_index}')
    print(f'window index is {reader.window_index}')
    print(f"first label is {label[0]}")
    print(f"second label is {label[1]}")