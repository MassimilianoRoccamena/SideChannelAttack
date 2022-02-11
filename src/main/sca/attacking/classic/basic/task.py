import os
import numpy as np
from tqdm.auto import trange

from sca.attacking.loader import *
from sca.attacking.classic.task import StaticDiscriminator

class BasicDiscriminator(StaticDiscriminator):
    '''
    Trace key discriminator on static frequency traces.
    '''

    def process_traces(self, voltage, frequency, key_value, traces):
        return traces

    def target_platform(self, voltage, frequency):
        return (voltage, frequency)