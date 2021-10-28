from core.window.slicer import AdvancedTraceSlicer
from core.window.classification.dataset import WindowClassificationDataset

from mining.data.loader import print_load

def test_window_classification_dataset():
    slicer = AdvancedTraceSlicer(1000, 50)
    dataset = WindowClassificationDataset(slicer, ["1.00"], ["52.000"],
                                            ["32","a3"], 500)

    window = dataset[0]
    print(window.shape)

if __name__ == "__main__":
    #print_load()
    test_window_classification_dataset()