from torch.utils.data import Dataset

from core.data.params import TRACE_LENGTH
from core.data.path import root_data_dir
from core.data.load import AdvancedLoader

class WindowClassificationDataset(Dataset):
    def __init__(self, voltages, frequencies, slicer):
        super().__init__()
        self.voltages = voltages
        self.frequencies = frequencies
        self.slicer = slicer

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass