import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import interp1d
import torch

from utils.persistence import save_numpy
from sca.loader import *
from sca.assembler import DynamicAssembler
from sca.profiling.classic.aligned.segmentation.task import GradCamSegmentation

class DynamicSegmentation(GradCamSegmentation):
    '''
    GRAD-CAM frequency segmentation from of static frequency traces.
    '''

    def __init__(self, loader, voltage, frequencies, key_values,
                    plain_bounds, training_path, checkpoint_file, batch_size, interp_kind,
                    mu, sigma, trace_len=None, log_segmentation=None, log_localization=None,
                    min_window_len=None, max_window_len=None, num_workers=None, workers_type=None):
        '''
        Create new GRAD-CAM frequency segmentation of dynamic traces.
        loader: power trace loader
        voltage: voltage of platform to segment
        frequencies: frequencies of platform to segment
        plain_bounds: start, end plain text indices
        training_path: root directory of a model training
        checkpoint_file: file name of the model checkpoint
        batch_size: batch size for model inference
        interp_kind: interpolation kind for map upscaling
        trace_len: size of the trace to segment
        log_segmentation: wheter to persist segmentation results
        log_localization: wheter to persist localization results
        min_window_len: min assembled window size inside a trace
        max_window_len: max assembled window size inside a trace
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        super().__init__(loader, [voltage], frequencies, key_values, \
                    plain_bounds, training_path, checkpoint_file, batch_size, \
                    interp_kind, trace_len, log_segmentation, log_localization, \
                    num_workers, workers_type)
        self.assembler = DynamicAssembler(loader, self.plain_indices, voltage, frequencies, \
                                mu, sigma, self.trace_len, min_window_len, max_window_len)

    def compute_work(self):
        '''
        Compute frequency segmentation + static window localization of the dynamic attack traces.
        '''
        segm_path = os.path.join(self.log_dir, 'segmentation')
        if self.log_segmentation:
            os.mkdir(segm_path)
        lclz_path = os.path.join(self.log_dir, 'localization')
        if self.log_localization:
            os.mkdir(lclz_path)

        print('Segmenting traces by frequencues\n')
        num_keys = len(self.key_values)
        num_workers = self.num_workers
        workers_type = self.workers_type

        if num_workers is None:
            pbar = tqdm.tqdm(total=num_keys)                                        # vanilla
            for key in self.key_values:
                segm = self.segmentation_work(key)
                if self.log_segmentation:
                    file_path = os.path.join(segm_path, f'{key}.npy')
                    save_numpy(segm, file_path)
                df_lclz = self.localization_work(key, segm)
                if self.log_localization:
                    file_path = os.path.join(lclz_path, f'{key}.csv')
                    df_lclz.to_csv(file_path, index=False)
                pbar.update(1)
        else:
            n_iters = ceil(len(self.key_values) / num_workers)                      # multiprocessed --- WIP
            pbar = tqdm.tqdm(total=n_iters)
            for i in range(n_iters):
                segm = Parallel(n_jobs=num_workers, prefer=workers_type) (delayed(self.segmentations_work) \
                                            (voltage, frequency, key) for key in self.key_values)
                save_numpy(segm, os.path.join(segm_path, f'WIP.npy'))
                pbar.update(1)
        pbar.close()