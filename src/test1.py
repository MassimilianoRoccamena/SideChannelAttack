# run tests for the main system
import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(PROJECT_PATH, "src/main")
sys.path.append(SOURCE_PATH)

import unittest
from testing.sca.file.convention1.loader import TraceLoader1Test
from testing.sca.profiling.window.classification.dataset \
    import StridedMultiClassification1Test

if __name__ == '__main__':
    unittest.main()