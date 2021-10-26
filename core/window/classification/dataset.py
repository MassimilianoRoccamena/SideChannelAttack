from torch.utils.data import Dataset

from core.data.params import TRACE_LENGTH

class WindowClassificationDataset(Dataset):
    def __init__(self, voltages, frequencies):
        super().__init__()
        self.voltages = voltages
        self.frequencies = frequencies

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass