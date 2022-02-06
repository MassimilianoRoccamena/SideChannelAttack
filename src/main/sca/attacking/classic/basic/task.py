import os
import numpy as np
from tqdm.auto import trange

from sca.attacking.discriminator.loader import *
from sca.attacking.discriminator.attacker import KeyAttacker

class BasicDiscriminator(KeyAttacker):
    '''
    Basic trace key attacker on power traces.
    '''

    def process_traces(self, voltage, frequency, key_value, traces):
        return traces

    def target_platform(self, voltage, frequency):
        return (voltage, frequency)