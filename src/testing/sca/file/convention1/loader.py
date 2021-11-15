import unittest

from main.sca.file.params import TRACE_SIZE
from main.sca.file.params import TEXT_SIZE
from main.sca.file.convention1.params import NUM_TRACES
from main.sca.file.convention1.loader import TraceLoader1
from main.sca.file.convention1.path import FileIdentifier, file_path

class TraceLoader1Test(unittest.TestCase):

    def setUp(self):
        file_id = FileIdentifier('1.00', '52.000', '00')
        self.loader = TraceLoader1(file_id)

    def test_all_shape(self):
        traces, texts = self.loader.load_all_traces()

        self.assertEqual(traces.shape, (NUM_TRACES, TRACE_SIZE),
                            'wrong traces shape')
        self.assertEqual(texts.shape, (NUM_TRACES, TEXT_SIZE),
                            'wrong texts shape')

    def test_some_shape(self):
        trace_idx = [0, 3, 99, 333, 499]
        traces, texts = self.loader.load_some_traces(trace_idx)

        self.assertEqual(traces.shape, (len(trace_idx), TRACE_SIZE),
                            'wrong traces shape')
        self.assertEqual(texts.shape, (len(trace_idx), TEXT_SIZE),
                            'wrong texts shape')

    def test_some_projected_shape(self):
        trace_idx = [40, 400]
        n = 10000
        time_idx = [i for i in range(100, 100+n)]
        traces, texts = self.loader.load_some_projected_traces(trace_idx, time_idx)
        size = (2, n)

        self.assertEqual(traces.shape, (len(trace_idx), n),
                            'wrong traces shape')
        self.assertEqual(texts.shape, (len(trace_idx), TEXT_SIZE),
                            'wrong texts shape')
