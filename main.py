from core.window.slicer import AdvancedTraceSlicer
from core.window.classification.dataset import WindowClassificationDataset

from mining.data.loader import print_load

if __name__ == "__main__":
    #print_load()
    slicer = AdvancedTraceSlicer(1000, 50)
    dataset = WindowClassificationDataset(slicer, ["1.00"], ["52.000"],
                                            ["32","a3"], 500)