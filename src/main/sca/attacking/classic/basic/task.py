import os
import numpy as np
from tqdm.auto import trange

from sca.attacking.classic.loader import *
from sca.attacking.classic.task import KeyDiscriminator

class BasicDiscriminator(KeyDiscriminator):
    '''
    Basic trace key discriminator on power traces.
    '''

    def process_traces(self, voltage, frequency, key_value, traces):
        return traces

    def target_platform(self, voltage, frequency):
        return (voltage, frequency)