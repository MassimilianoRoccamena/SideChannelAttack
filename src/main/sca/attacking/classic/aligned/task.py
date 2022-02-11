import os
import numpy as np
from tqdm.auto import trange

from sca.rescaler import FrequencyRescaler
from sca.attacking.loader import *
from sca.attacking.classic.task import StaticDiscriminator

class AlignedStaticDiscriminator(StaticDiscriminator):
    '''
    Trace key discriminator on aligned static frequency traces.
    '''

    def __init__(self, generator_path, voltages, frequencies,
                        plain_bounds, num_workers, workers_type,
                        target_volt, target_freq, interp_kind=None):
        '''
        Create new template attacker on aligned static traces.
        generator_path: path of a trace generator
        voltages: voltages of platform to attack
        frequencies: frequencies of platform to attack
        plain_bounds: start, end plain text indices
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        target_freq: voltage of the aligned platform
        target_freq: frequency of the aligned platform
        interp_kind: kind of 1D trace interpolation
        '''
        super().__init__(generator_path, voltages, frequencies,
                            plain_bounds, num_workers, workers_type)
        self.target_volt = target_volt
        self.target_freq = target_freq
        if interp_kind is None:
            interp_kind = 'linear'
        self.interp_kind = interp_kind

    def process_traces(self, voltage, frequency, key_value, traces):
        freq_ratio = float(self.target_freq) / float(frequency)
        rescaler = FrequencyRescaler(freq_ratio, self.interp_kind)
        return rescaler.scale_windows(traces)

    def target_platform(self, voltage, frequency):
        return (self.target_volt, self.target_freq)