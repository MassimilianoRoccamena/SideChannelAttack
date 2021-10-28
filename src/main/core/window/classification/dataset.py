from torch.utils.data import Dataset

from src.main.core.window.reader import WindowReader

class AbstractWindowClassification(WindowReader, Dataset):
    '''
    Abstract dataset which loads power trace windows with some labels
    '''

    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        super().__init__(slicer, voltages, frequencies, key_values, num_traces)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        raise NotImplementedError

class MixedWindowClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with voltage and frequency
    '''

    def __getitem__(self, index):
        x = super().read_window(index)
        y0 = self.voltages.index(self.file_id.voltage)
        y1 = self.frequencies.index(self.file_id.frequency)
        return x, (y0, y1)

class VoltageWindowClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with voltage
    '''

    def __getitem__(self, index):
        x = super().__getitem__(index)
        y = self.voltages.index(self.file_id.voltage)
        return x, y

class FrequencyClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    def __getitem__(self, index):
        x = super().__getitem__(index)
        y = self.frequencies.index(self.file_id.frequency)
        return x, y