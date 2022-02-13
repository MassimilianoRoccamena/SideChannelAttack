import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
import torch

from utils.persistence import save_numpy
from sca.loader import *
from sca.assembler import DynamicAssembler
from sca.profiling.classic.aligned.segmentation.task import GradCamSegmentation

class DynamicSegmentation(GradCamSegmentation):
    '''
    GRAD-CAM frequency segmentation from of static frequency traces.
    '''

    def __init__(self, loader, training_path, checkpoint_file,
                    key_values, plain_bounds, batch_size, interp_kind,
                    mu, sigma, log_assembler=None, log_segmentation=None, log_localization=None,
                    min_window_len=None, max_window_len=None, num_workers=None, workers_type=None):
        '''
        Create new GRAD-CAM frequency segmentation of dynamic traces.
        loader: power trace loader
        training_path: root directory of a model training
        checkpoint_file: file name of the model checkpoint
        plain_bounds: start, end plain text indices
        batch_size: batch size for model inference
        interp_kind: interpolation kind for map upscaling
        mu: mean of the gaussian duration of the static windows
        sigma: std of the gaussian duration of the static windows
        log_assembler: wheter to persist assembler results
        log_segmentation: wheter to persist segmentation results
        log_localization: wheter to persist localization results
        min_window_len: min static window size
        max_window_len: max static window size
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        super().__init__(loader, training_path, checkpoint_file, key_values, \
                        plain_bounds, batch_size, interp_kind, log_assembler, \
                        log_segmentation, log_localization, \
                        num_workers, workers_type)
        self.assembler = DynamicAssembler(loader, self.plain_indices, self.voltages[0], self.frequencies, \
                                mu, sigma, min_window_len, max_window_len, track_windows=log_assembler)

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