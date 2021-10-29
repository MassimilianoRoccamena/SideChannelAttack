# mine informations about something in the overall system

from main.core.window.slicer import AdvancedTraceSlicer
from main.core.window.classification.dataset import MixedWindowClassification

from mining.base.data.loader import print_loader_example
from mining.core.window.classification.dataset import print_window_classification_item

if __name__ == "__main__":
    print_loader_example()
    print("")
    print_window_classification_item()