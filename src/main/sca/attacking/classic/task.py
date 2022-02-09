import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import multivariate_normal

from utils.persistence import load_pickle, load_json, save_json, load_numpy, save_numpy
from utils.math import BYTE_SIZE, BYTE_HW_LEN, pca_transform
from aidenv.api.config import get_program_log_dir
from aidenv.api.basic.config import build_task_kwarg
from aidenv.api.mlearn.task import MachineLearningTask
from sca.file.params import SBOX_MAT, HAMMING_WEIGHTS

class KeyDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some attack traces using
    a fitted reduced trace generative model.
    '''

    def __init__(self, loader, generator_path, voltages, frequencies, plain_bounds,
                    num_workers, workers_type):
        '''
        Create new PCA+QDA template discriminator.
        loader: power trace loader
        generator_path: path of a trace generator
        voltages: voltages of platform to attack
        frequencies: frequencies of platform to attack
        plain_bounds: start, end plain text indices
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.loader = loader
        self.generator_path = generator_path
        generator_params = load_json(os.path.join(generator_path, 'params.json'))
        if voltages is None:
            self.voltages = generator_params['voltages']
        else:
            self.voltages = list(voltages)
        if frequencies is None:
            self.frequencies = generator_params['frequencies']
        else:
            self.frequencies = list(frequencies)
        self.key_values = generator_params['key_values']
        self.hw_len = BYTE_HW_LEN
        self.plain_bounds = list(plain_bounds)
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.reduced_dim = generator_params['reduced_dim']
        self.num_workers = num_workers
        self.workers_type = workers_type

    @classmethod
    @build_task_kwarg('loader')
    def build_kwargs(cls, config):
        pass

    def process_traces(self, voltage, frequency, key_value, traces):
        '''
        Process loaded traces for the computation of one iteration of key likelihoods.
        '''
        raise NotImplementedError
    def target_platform(self, voltage, frequency):
        '''
        Select target (voltage,frequency) platform to be used to compute likelihoods.
        '''
        raise NotImplementedError

    def key_likelihoods_work(self, pca, gauss_mean, gauss_cov,
                                    voltage, frequency, key_true, keys_lh):
        '''
        Work method of one process computing one true key likelihoods over all key hypothesis.
        '''
        num_keys = len(self.key_values)

        file_id = self.loader.build_file_id(voltage, frequency, key_true)
        file_path = self.loader.build_file_path(file_id)
        traces, plain_texts, key = self.loader.load_some_traces(file_path, self.plain_indices)

        traces = self.process_traces(voltage, frequency, key_true, traces)
        traces = pca_transform(pca, traces)
            
        for plain_idx in range(self.num_plain_texts):
            curr_trace = traces[plain_idx]

            for key_hyp in range(num_keys):
                plain_text = plain_texts[plain_idx][0]
                sbox_hw = HAMMING_WEIGHTS[SBOX_MAT[plain_text ^ key_hyp]]

                multi_gauss = multivariate_normal(gauss_mean[sbox_hw], gauss_cov[sbox_hw])
                prob_hyp = multi_gauss.pdf(curr_trace)
                    
                keys_lh[int(f'0x{key_true}', base=16), key_hyp, plain_idx:] -= np.log(prob_hyp)

    def compute_likelihoods(self, voltage, frequency):
        '''
        Compute likelihoods of keys for the attack traces of the (voltage,frequency) platform.
        '''
        voltage_target, frequency_target = self.target_platform(voltage, frequency)
        root_path = os.path.join(self.generator_path, f'{voltage_target}-{frequency_target}')
        curr_path = os.path.join(root_path, 'pca')
        pca = load_pickle(os.path.join(curr_path, 'pca.pckl'))
        curr_path = os.path.join(root_path, 'multi_gauss')
        gauss_mean = load_numpy(os.path.join(curr_path, 'mean.npy'))
        gauss_cov = load_numpy(os.path.join(curr_path, 'covariance.npy'))

        num_keys = len(self.key_values)
        keys_lh = np.zeros((num_keys, num_keys, self.num_plain_texts))

        num_workers = self.num_workers
        workers_type = self.workers_type

        if num_workers is None:
            pbar = tqdm.tqdm(total=num_keys)                                        # vanilla
            for key_true in self.key_values:
                self.key_likelihoods_work(pca, gauss_mean, gauss_cov, voltage,
                                            frequency, key_true, keys_lh)
                pbar.update(1)
        else:
            n_iters = ceil(len(self.key_values) / num_workers)                      # multiprocessed
            pbar = tqdm.tqdm(total=n_iters)
            for i in range(n_iters):
                keys_true = self.key_values[i : min((i+1)*num_workers, num_keys-1)]
                Parallel(n_jobs=num_workers, prefer=workers_type) (delayed(self.key_likelihoods_work) \
                        (pca, gauss_mean, gauss_cov, voltage, frequency, key_true, keys_lh) for key_true in keys_true)
                pbar.update(1)

        pbar.close()
        return keys_lh

    def run(self, *args):
        log_dir = get_program_log_dir()

        # init params
        params = {'generator_path':self.generator_path,
                    'plain_bounds':self.plain_bounds}

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

        # save params
        params_path = os.path.join(log_dir, 'params.json')
        save_json(params, params_path)