from torch.utils.data import Dataset

class AbstractWindowClassification(Dataset):
    '''
    Abstract dataset which reads power trace windows with some labels
    '''

    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    def __len__(self):
        return len(self.reader.slicer)

    def __getitem__(self, index):
        raise NotImplementedError

class MixedWindowClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with voltage and frequency
    '''

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y0 = reader.voltages.index(reader.file_id.voltage)
        y1 = reader.frequencies.index(reader.file_id.frequency)
        return x, (y0, y1)

class VoltageWindowClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with voltage
    '''

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y = reader.voltages.index(reader.file_id.voltage)
        return x, y

class FrequencyWindowClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y = reader.frequencies.index(reader.file_id.frequency)
        return x, y
