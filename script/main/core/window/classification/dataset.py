from main.base.app.config import config_core_object1
from main.core.dataset import ClassificationDataset
from main.core.window.params import SLICER_MODULE
from main.core.window.reader import WindowReader

class WindowClassification(ClassificationDataset):
    '''
    Abstract dataset which reads power trace windows with some labels
    '''

    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        '''
        Create new window classification dataset.
        slicer: window slicing strategy
        voltages: desired voltages
        frequencies: desired frequencies
        key_values: desired key values
        num_traces: number of traces in each file
        '''
        super().__init__()
        self.reader = WindowReader(slicer, voltages, frequencies,
                                    key_values, num_traces)

    @classmethod
    def config_args(cls, config, core_nodes):
        slicer = config_core_object1(config.slicer, core_nodes[:1],
                                        SLICER_MODULE)

        return [ slicer, config.voltages, config.frequencies,
                 config.key_values, config.num_traces ]

    def __len__(self):
        return len(self.reader.slicer)

    def __getitem__(self, index):
        raise NotImplementedError

class MultiClassification(WindowClassification):
    '''
    Dataset composed of power trace windows with voltage
    and frequency labelling
    '''

    def get_num_classes(self):
        return ( len(self.voltages),
                 len(self.frequencies) )

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y0 = reader.voltages.index(reader.file_id.voltage)
        y1 = reader.frequencies.index(reader.file_id.frequency)
        return x, (y0, y1)

class VoltageClassification(WindowClassification):
    '''
    Dataset composed of power trace windows labelled with voltage
    '''

    def get_num_classes(self):
        return len(self.voltages)

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y = reader.voltages.index(reader.file_id.voltage)
        return x, y

class FrequencyClassification(WindowClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    def get_num_classes(self):
        return len(self.frequencies)

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y = reader.frequencies.index(reader.file_id.frequency)
        return x, y
