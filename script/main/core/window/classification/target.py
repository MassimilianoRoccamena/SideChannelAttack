from main.core.window.slicer import AdvancedTraceSlicer
from main.core.window.reader import WindowReader
from main.core.window.classification.dataset import MixedWindowClassification
from main.core.window.classification.dataset import VoltageWindowClassification
from main.core.window.classification.dataset import FrequencyWindowClassification
from main.core.window.classification.model import get_A

# dataset

WINDOW_SIZE = 1000
STRIDE = 50

VOLTAGES = ['1.00']
FREQUENCIES = ['52.000']
KEY_VALUES = ['32','a3']
NUM_TRACES = 500

def get_reader():
    slicer = AdvancedTraceSlicer(WINDOW_SIZE, STRIDE)
    return WindowReader(slicer, VOLTAGES, FREQUENCIES, KEY_VALUES, NUM_TRACES)

def get_mixed_labels_dataset():
    return MixedWindowClassification(get_reader())

def get_voltage_labels_dataset():
    return VoltageWindowClassification(get_reader())

def get_frequency_labels_dataset():
    return FrequencyWindowClassification(get_reader())

def get_dataset():
    return get_frequency_labels_dataset()

# model

def get_model():
    return get_A()