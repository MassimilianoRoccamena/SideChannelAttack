from torch.utils.data import Dataset

from core.window.reader import WindowReader

class WindowClassificationDataset(WindowReader, Dataset):
    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        super().__init__(slicer, voltages, frequencies, key_values, num_traces)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        x = super().__getitem__(index)
        l0 = self.voltages.index(self.file_id.voltage)
        l1 = self.frequencies.index(self.file_id.frequency)
        return x, (l0, l1)