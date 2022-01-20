import os
import numpy as np
from tqdm.auto import trange

from utils.persistence import save_json
from aidenv.api.config import get_program_log_dir
from aidenv.api.basic.config import build_task_kwarg
from aidenv.api.dprocess.task import DataProcess
from sca.preprocessing.window.loader import WindowLoader1 as FileConvention1
from sca.preprocessing.window.loader import WindowLoader2 as FileConvention2

class TraceInterpolator(DataProcess):
    pass