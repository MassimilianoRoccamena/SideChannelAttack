from torch.utils.data import Dataset

from core.data.path import root_data_dir
from core.data.load import AdvancedLoader

class WindowClassificationDataset(Dataset):
    def __init__(self, voltages, frequencies, window, stride=None):
        super().__init__()
        self.voltages = voltages
        self.frequencies = frequencies
        self.window = window
        if stride is None:
            self.stride = window
        else:
            self.stride = stride

        self.search_items()

    def search_items(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass