from main.base.app.params import DATASET_MODULE
from main.base.app.config import build_core_object1
from main.core.dataset import ClassificationDataset
from main.core.window.slicer import StridedSlicer
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
    def build_args(cls, config, core_nodes):
        slicer = build_core_object1(config.slicer, core_nodes, DATASET_MODULE)

        return [ slicer, config.voltages, config.frequencies,
                 config.key_values, config.num_traces ]

    def __len__(self):
        return len(self.reader.slicer)

class SingleClassification(WindowClassification):
    '''
    Abstract dataset composed of power trace windows with one label
    '''

    def __getitem__(self, index):
        x = self.reader[index]
        labels = self.all_labels()
        label = self.current_label()
        y = labels.index(label)
        return x, y

class MultiClassification(WindowClassification):
    '''
    Dataset composed of power trace windows with (voltage, frequency)
    labelling
    '''

    def all_labels(self):
        return ( self.reader.voltages,
                 self.reader.frequencies )

    def current_label(self):
        return ( self.reader.file_id.voltage,
                 self.reader.file_id.frequency )

    def __getitem__(self, index):
        x = self.reader[index]
        labels = self.all_labels()
        label = self.current_label()
        y0 = labels[0].index(label[0])
        y1 = labels[1].index(label[1])
        return x, (y0, y1)

class VoltageClassification(SingleClassification):
    '''
    Dataset composed of power trace windows labelled with voltage
    '''

    def all_labels(self):
        return self.reader.voltages

    def current_label(self):
        return self.reader.file_id.voltage

class FrequencyClassification(WindowClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    def all_labels(self):
        return self.reader.frequencies

    def current_label(self):
        return self.reader.file_id.frequency