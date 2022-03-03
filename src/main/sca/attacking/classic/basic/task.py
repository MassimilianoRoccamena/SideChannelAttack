import os
import numpy as np
from tqdm.auto import trange

from sca.attacking.loader import *
from sca.attacking.classic.task import StaticDiscriminator

class BasicDiscriminator(StaticDiscriminator):
    '''
    Basic trace key discriminator on static frequency traces.
    '''

    def __init__(self, generator_path, voltages, frequencies, key_values,
                    plain_bounds, num_workers=None, workers_type=None):
        '''
        Create new basic template attacker on static traces.
        generator_path: path of a trace generator
        voltages: voltages of platforms to attack
        frequencies: frequencies of platform to attack
        key_values: key values to attack
        plain_bounds: start, end plain text indices
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        super().__init__(generator_path, voltages, frequencies, key_values,
                            plain_bounds, num_workers, workers_type)

    def process_traces(self, voltage, frequency, key_value, traces):
        return traces

    def target_platform(self, voltage, frequency):
        return (voltage, frequency)