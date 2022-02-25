import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import multivariate_normal

from utils.persistence import load_pickle, load_numpy, save_numpy
from utils.math import BYTE_SIZE, BYTE_HW_LEN, pca_transform
from aidenv.api.config import get_program_log_dir
from aidenv.api.mlearn.task import MachineLearningTask
from sca.config import OmegaConf, build_task_object
from sca.file.params import str_hex_bytes, SBOX_MAT, HAMMING_WEIGHTS

class StaticDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some static
    frequency traces using a fitted reduced trace generative model.
    '''

    def __init__(self, generator_path, voltages, frequencies, plain_bounds,
                    num_workers, workers_type):
        '''
        Create new template attacker on static traces.
        generator_path: path of a trace generator
        voltages: voltages of platforms to attack
        frequencies: frequencies of platform to attack
        plain_bounds: start, end plain text indices
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.generator_path = generator_path
        generator_path = os.path.join(generator_path, 'program.yaml')
        self.generator_config = OmegaConf.to_object(OmegaConf.load(generator_path))
        self.loader = build_task_object(self.generator_config['core']['params']['loader'])
        if voltages is None:
            self.voltages = self.generator_config['core']['params']['voltages']
            print(f'Found {len(self.voltages)} voltages')
        else:
            self.voltages = voltages
        if frequencies is None:
            self.frequencies = self.generator_config['core']['params']['frequencies']
            print(f'Found {len(self.frequencies)} frequencies')
        else:
            self.frequencies = frequencies
        self.key_values = self.generator_config['core']['params']['key_values']
        if self.key_values is None:
            self.key_values = str_hex_bytes()
            print('Using all key values')
        self.plain_bounds = plain_bounds
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.reduced_dim = self.generator_config['core']['params']['reduced_dim']
        self.num_workers = num_workers
        self.workers_type = workers_type
        if not self.num_workers is None:
            raise NotImplementedError('multiprocessing WIP')
        self.log_dir = get_program_log_dir()

    def process_traces(self, voltage, frequency, key_value, traces):
        '''
        Process loaded traces for the computation of one iteration of key likelihoods.
        '''
        raise NotImplementedError
    def target_platform(self, voltage, frequency):
        '''
        Select which templates of which target (voltage,frequency) platform.
        '''
        raise NotImplementedError

    def key_likelihood_work(self, pca, gauss_mean, gauss_cov,
                                    voltage, frequency, key_true):
        '''
        Work method of one process computing one true key likelihoods over all key hypothesis.
        '''
        num_keys = len(self.key_values)
        key_lh = np.zeros((num_keys, self.num_plain_texts))

        file_id = self.loader.build_file_id(voltage, frequency, key_true)
        file_path = self.loader.build_file_path(file_id)
        traces, plain_texts, key = self.loader.fetch_traces(file_path, self.plain_indices)

        processed_traces = self.process_traces(voltage, frequency, key_true, traces)
        original_len = traces.shape[-1]
        processed_len = processed_traces.shape[-1]

        if processed_len > original_len:
            traces = processed_traces[:,:original_len]
        elif processed_len < original_len:
            traces = np.zeros(traces.shape)
            traces[:,:processed_len] = processed_traces[:,:]
        traces = pca_transform(pca, traces)
            
        for plain_idx in range(self.num_plain_texts):
            curr_trace = traces[plain_idx]

            for key_hyp in range(num_keys):
                plain_text = plain_texts[plain_idx][0]
                sbox_hw = HAMMING_WEIGHTS[SBOX_MAT[plain_text ^ key_hyp]]

                multi_gauss = multivariate_normal(gauss_mean[sbox_hw], gauss_cov[sbox_hw])
                prob_hyp = multi_gauss.pdf(curr_trace)
                    
                key_lh[key_hyp, plain_idx:] -= np.log(prob_hyp)

        return key_lh

    def compute_work(self, voltage, frequency):
        '''
        Compute likelihoods of keys for the attack traces of the (voltage,frequency) platform.
        '''
        voltage_target, frequency_target = self.target_platform(voltage, frequency)
        template_path = os.path.join(self.generator_path, f'{voltage_target}-{frequency_target}')
        curr_path = os.path.join(template_path, 'pca')
        pca = load_pickle(os.path.join(curr_path, 'pca.pckl'))
        curr_path = os.path.join(template_path, 'multi_gauss')
        gauss_mean = load_numpy(os.path.join(curr_path, 'mean.npy'))
        gauss_cov = load_numpy(os.path.join(curr_path, 'covariance.npy'))
        num_keys = len(self.key_values)

        num_workers = self.num_workers
        workers_type = self.workers_type

        if num_workers is None:
            pbar = tqdm.tqdm(total=num_keys)                                        # vanilla
            for key_true in self.key_values:
                key_lh = self.key_likelihood_work(pca, gauss_mean, gauss_cov, voltage, frequency, key_true)
                file_path = os.path.join(self.lh_path, f'{key_true}.npy')
                save_numpy(key_lh, file_path)
                pbar.update(1)
        else:
            n_iters = ceil(len(self.key_values) / num_workers)                      # multiprocessed -- WIP
            pbar = tqdm.tqdm(total=n_iters)
            for i in range(n_iters):
                keys_true = self.key_values[i : min((i+1)*num_workers, num_keys-1)]
                Parallel(n_jobs=num_workers, prefer=workers_type) (delayed(self.key_likelihoods_work) \
                        (pca, gauss_mean, gauss_cov, voltage, frequency, key_true, keys_lh) for key_true in keys_true)
                pbar.update(1)

        pbar.close()

    def run(self, *args):
        for voltage in self.voltages:
            for frequency in self.frequencies:
                print(f'\nProcessing {voltage}-{frequency} platform')
                platform_path = os.path.join(self.log_dir, f'{voltage}-{frequency}')
                os.mkdir(platform_path)
                self.lh_path = platform_path

                print('Computing keys likelihood\n')
                self.compute_work(voltage, frequency)
