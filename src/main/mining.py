# mine informations about something in the main system

from sca.file.loader.mining import print_loader_example
from sca.core.window.classification.dataset.mining \
    import print_window_classification_item

if __name__ == '__main__':
    print("LOADED DATA SHAPES\n")
    print_loader_example()
    print("\nWINDOW CLASSIFICATION DATASET\n")
    print_window_classification_item()