from main.base.launcher.config import ConfigParser
from main.core.dataset import ConfigDataset
from main.core.window.reader import WindowReader

class AbstractWindowClassification(ConfigDataset):
    '''
    Abstract dataset which reads power trace windows with some labels
    '''

    class Parser(ConfigParser):
        '''
        Parser from config of an abstact dataset of trace windows
        '''

        def constr_args(self, config):
            return [ config.slicer,
                     config.voltages,
                     config.frequencies,
                     config.key_values,
                     config.num_traces ]

    def __init__(self, parser, slicer, voltages, frequencies, key_values, num_traces):
        '''
        Create new window classification dataset.
        slicer: window slicing strategy
        voltages: desired voltages
        frequencies: desired frequencies
        key_values: desired key values
        num_traces: number of traces in each file
        '''
        super().__init__(parser)
        self.reader = WindowReader(slicer, voltages, frequencies, key_values, num_traces)

    def __len__(self):
        return len(self.reader.slicer)

    def __getitem__(self, index):
        raise NotImplementedError

class MixedWindowClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with voltage and frequency
    '''

    class Parser(AbstractWindowClassification.Parser):
        '''
        Abstract parser from config of a mixed classification dataset
        '''

        def __init__(self):
            super().__init__(MixedWindowClassification)

    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        super().__init__( MixedWindowClassification.Parser(),
                          slicer, voltages, frequencies,
                          key_values, num_traces )

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

    class Parser(AbstractWindowClassification.Parser):
        '''
        Abstract parser from config of a voltage classification dataset
        '''
        
        def __init__(self):
            super().__init__(VoltageWindowClassification)

    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        super().__init__( VoltageWindowClassification.Parser(),
                          slicer, voltages, frequencies,
                          key_values, num_traces )

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y = reader.voltages.index(reader.file_id.voltage)
        return x, y

class FrequencyWindowClassification(AbstractWindowClassification):
    '''
    Dataset composed of power trace windows labelled with frequency
    '''

    class Parser(AbstractWindowClassification.Parser):
        '''
        Abstract parser from config of a frequency classification dataset
        '''
        
        def __init__(self):
            super().__init__(FrequencyWindowClassification)

    def __init__(self, slicer, voltages, frequencies, key_values, num_traces):
        super().__init__( FrequencyWindowClassification.Parser(),
                          slicer, voltages, frequencies,
                          key_values, num_traces )

    def __getitem__(self, index):
        reader = self.reader
        x = reader[index]
        y = reader.frequencies.index(reader.file_id.frequency)
        return x, y
