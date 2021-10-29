from main.core.window.classification.target import get_mixed_labels_dataset

def print_window_classification_item():
    dataset = get_mixed_labels_dataset()
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