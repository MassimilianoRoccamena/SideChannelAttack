from main.core.window.slicer import AdvancedTraceSlicer
from main.core.window.reader import WindowReader
from main.core.window.classification.dataset import MixedWindowClassification

def print_window_classification_item():
    slicer = AdvancedTraceSlicer(1000, 50)
    reader = WindowReader(slicer, ['1.00'], ['52.000'], ['32','a3'], 500)
    dataset = MixedWindowClassification(reader)

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