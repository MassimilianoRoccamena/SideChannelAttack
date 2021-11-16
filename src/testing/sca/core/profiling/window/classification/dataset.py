import unittest
import numpy as np

from main.sca.file.convention1.params import NUM_TRACES
from main.sca.core.profiling.window.loader import WindowLoader1
from main.sca.core.profiling.window.slicer import StridedSlicer
from main.sca.core.profiling.window.classification.dataset \
    import MultiClassification

class StridedMultiClassification1Test(unittest.TestCase):
    
    def setUp(self):
        self.window_size = 1000
        self.stride = 50
        slicer = StridedSlicer(self.window_size, self.stride)
        loader = WindowLoader1(slicer)
        self.dataset = MultiClassification(loader,
                                            ['1.00'], ['48.000','52.000'],
                                            ['00','01'], NUM_TRACES)

    def test_random_reads(self):
        indices = np.random.randint(0, len(self.dataset), size=100)

        for idx in indices:
            window, label = self.dataset[0]

            self.assertEqual(window.shape, (1,self.window_size),
                                'wrong texts shape')

            ok_label = label == (0,0) or label == (0,1)
            self.assertTrue(ok_label, 'wrong texts shape')