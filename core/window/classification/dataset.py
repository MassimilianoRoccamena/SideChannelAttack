from torch.utils.data import Dataset

from core.data.params import TRACE_SIZE

class WindowClassificationDataset(Dataset):
    def __init__(self, volt, freq):
        super().__init__()
        self.voltages = volt
        self.frequencies = freq

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass