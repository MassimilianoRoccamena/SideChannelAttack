from src.main.core.window.slicer import AdvancedTraceSlicer
from src.main.core.window.classification.dataset import MixedWindowClassification

def print_window_classification_item():
    slicer = AdvancedTraceSlicer(1000, 50)
    dataset = MixedWindowClassification(slicer, ['1.00'], ['52.000'],
                                            ['32','a3'], 500)

    reader_idx = 1400000
    window, label = dataset[reader_idx]

    print(f'dataset index is {reader_idx}')
    print(f'window shape is {window.shape}')
    print(f'voltage is {dataset.file_id.voltage}')
    print(f'frequency is {dataset.file_id.frequency}')
    print(f'key value is {dataset.file_id.key_value}')
    print(f'trace index is {dataset.trace_index}')
    print(f'window index is {dataset.window_index}')
    print(f"first label is {label[0]}")
    print(f"second label is {label[1]}")