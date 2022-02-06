import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import multivariate_normal
import torch

from utils.persistence import save_numpy
from utils.math import BYTE_SIZE, BYTE_HW_LEN, pca_transform
from aidenv.api.config import get_program_log_dir
from aidenv.api.basic.config import build_task_kwarg
from aidenv.api.mlearn.task import MachineLearningTask
from sca.file.params import SBOX_MAT, HAMMING_WEIGHTS
from sca.file.params import str_hex_bytes
from sca.attacking.deep.loader import *
from sca.attacking.deep.config import build_model_object, OmegaConf

class DeepDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some attack traces using
    a fitted deep network for sbox hw classification.
    '''

    def __init__(self, loader, voltages, frequencies, key_values, plain_bounds,
                    training_path, checkpoint_file, batch_size, num_workers=None, workers_type=None):
        '''
        Create new deep key attacker.
        loader: trace windows loader
        generator_path: path of a trace generator
        voltages: voltages of platform to attack
        frequencies: frequencies of platform to attack
        plain_bounds: start, end plain text indices
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.loader = loader
        self.voltages = list(voltages)
        self.frequencies = list(frequencies)
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        self.key_values = list(key_values)
        self.plain_bounds = list(plain_bounds)
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.training_path = training_path
        self.checkpoint_file = checkpoint_file
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.workers_type = workers_type

    @classmethod
    @build_task_kwarg('loader')
    def build_kwargs(cls, config):
        pass

    def key_likelihoods_work(self, voltage, frequency, key_true, keys_lh):
        '''
        Work method of one process computing one true key likelihoods over all key hypothesis.
        '''
        num_keys = len(self.key_values)

        file_id = self.loader.build_file_id(voltage, frequency, key_true)
        file_path = self.loader.build_file_path(file_id)
        traces, plain_texts, key = self.loader.load_some_traces(file_path, self.plain_indices)
        
        for key_hyp in range(BYTE_SIZE):
            n_iters = ceil(self.num_plain_texts / self.batch_size)

            for i in range(n_iters):
                low_plain_idx = i*self.batch_size
                high_plain_idx = min((i+1)*self.batch_size, self.num_plain_texts-1)
                real_batch_size = high_plain_idx - low_plain_idx
                curr_traces = traces[low_plain_idx : high_plain_idx]
                curr_traces = curr_traces.reshape(real_batch_size, 1, traces.shape[-1])
                curr_traces = torch.from_numpy(curr_traces)
                curr_traces = curr_traces.to(self.device)
                y_hat = self.model(curr_traces)

                for j in range(real_batch_size):
                    plain_idx = i*self.batch_size + j
                    plain_text = plain_texts[plain_idx][0]
                    sbox_hw = HAMMING_WEIGHTS[SBOX_MAT[plain_text ^ key_hyp]]
                    prob_hyp = y_hat[j, sbox_hw].detach().cpu().numpy()
                    key_true_idx = self.key_values.index(key_true)
                    keys_lh[key_true_idx, key_hyp, plain_idx:] -= np.log(prob_hyp)

    def compute_likelihoods(self, voltage, frequency):
        '''
        Compute likelihoods of keys for the attack traces of the (voltage,frequency) platform.
        '''
        num_keys = len(self.key_values)
        keys_lh = np.zeros((num_keys, BYTE_SIZE, self.num_plain_texts))

        num_workers = self.num_workers
        workers_type = self.workers_type

        if num_workers is None:
            pbar = tqdm.tqdm(total=num_keys)                                        # vanilla
            for key_true in self.key_values:
                self.key_likelihoods_work(voltage, frequency, key_true, keys_lh)
                pbar.update(1)
        else:
            n_iters = ceil(len(self.key_values) / num_workers)                      # multiprocessed
            pbar = tqdm.tqdm(total=n_iters)
            for i in range(n_iters):
                keys_true = self.key_values[i : min((i+1)*num_workers, num_keys-1)]
                Parallel(n_jobs=num_workers, prefer=workers_type) (delayed(self.key_likelihoods_work) \
                        (voltage, frequency, key_true, keys_lh) for key_true in keys_true)
                pbar.update(1)

        pbar.close()
        return keys_lh

    def run(self, *args):
        log_dir = get_program_log_dir()

        training_path = os.path.join(self.training_path, 'program.yaml')
        training_config = OmegaConf.load(training_path)
        model = build_model_object(training_config.model)
        print('Loaded model')

        labels = [str(i) for i in range(BYTE_HW_LEN)]
        model.module.set_labels(labels)
        checkpoint_path = os.path.join(self.training_path, 'checkpoints', self.checkpoint_file)
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        del checkpoint['loss.weight']
        model.load_state_dict(checkpoint)
        model.to(self.device)
        self.model = model
        print('Loaded model checkpoint')

        # platforms
        for voltage in self.voltages:
            for frequency in self.frequencies:
                print(f'\nProcessing {voltage}-{frequency} platform')
                curr_root_path = os.path.join(log_dir, f'{voltage}-{frequency}')
                if not os.path.exists(curr_root_path):
                    os.mkdir(curr_root_path)

                # likelihoods
                print('Computing keys likelihood\n')
                keys_lh = self.compute_likelihoods(voltage, frequency)
                save_numpy(keys_lh, os.path.join(curr_root_path, 'keys_likelihoods.npy'))
