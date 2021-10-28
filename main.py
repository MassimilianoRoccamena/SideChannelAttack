from core.window.slicer import AdvancedTraceSlicer
from core.window.classification.dataset import WindowClassificationDataset

from mining.base.loader import print_load

def test_window_classification_dataset():
    slicer = AdvancedTraceSlicer(1000, 50)
    dataset = WindowClassificationDataset(slicer, ['1.00'], ['52.000'],
                                            ['32','a3'], 500)

    window, label = dataset[300000]
    print(f'window shape is {window.shape}')
    print(f'voltage is {dataset.file_id.voltage}')
    print(f'frequency is {dataset.file_id.frequency}')
    print(f'key value is {dataset.file_id.key_value}')
    print(f'trace index is {dataset.trace_index}')
    print(f'window index is {dataset.window_index}')
    print(f"first label is {label[0]}")
    print(f"second label is {label[1]}")

if __name__ == "__main__":
    #print_load()
    test_window_classification_dataset()