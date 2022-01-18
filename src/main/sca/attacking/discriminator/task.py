import os
import tqdm
import math
import numpy as np
from scipy.stats import multivariate_normal

from utils.persistence import load_pickle, load_json, save_json, load_numpy, save_numpy
from utils.math import BYTE_SIZE, pca_transform
from aidenv.api.config import get_program_log_dir
from aidenv.api.basic.config import build_task_kwarg
from aidenv.api.mlearn.task import MachineLearningTask
from sca.file.params import TRACE_SIZE, str_hex_bytes, SBOX_MAT, HAMMING_WEIGHTS
from sca.file.convention1.loader import TraceLoader1 as FileConvention1
from sca.file.convention2.loader import TraceLoader2 as FileConvention2

class KeyDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some attack traces using
    a fitted reduced trace generative model.
    '''

    def __init__(self, loader, generator_path, plain_bounds):
        '''
        Create new PCA+QDA template attacker.
        loader: trace windows loader
        generator_path: path of a trace generator
        log_dir: log directory of the task
        '''
        self.loader = loader
        self.generator_path = generator_path
        generator_params = load_json(os.path.join(generator_path, 'params.json'))
        self.voltages = generator_params['voltages']
        self.frequencies = generator_params['voltages']
        self.key_values = generator_params['key_values']
        self.hw_len = math.ceil(math.log2(len(self.key_values))) + 1
        self.plain_bounds = list(plain_bounds)
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.reduced_dim = generator_params['reduced_dim']

    @classmethod
    @build_task_kwarg('loader')
    def build_kwargs(cls, config):
        pass

    def compute_likelihoods(self, voltage, frequency):
        root_path = os.path.join(self.generator_path, f'{voltage}-{frequency}')
        curr_path = os.path.join(root_path, 'pca')
        pca = load_pickle(os.path.join(curr_path, 'pca.pckl'))
        curr_path = os.path.join(root_path, 'multi_gauss')
        gauss_mean = load_numpy(os.path.join(curr_path, 'mean.npy'))
        gauss_cov = load_numpy(os.path.join(curr_path, 'covariance.npy'))

        num_keys = len(self.key_values)
        keys_lh = np.zeros((num_keys, num_keys, self.num_plain_texts))
        pbar = tqdm.tqdm(total=num_keys)

        for key_true in self.key_values:
            file_id = self.loader.build_file_id(voltage, frequency, key_true)
            file_path = self.loader.build_file_path(file_id)
            traces, plain_texts, key = self.loader.load_some_traces(file_path, self.plain_indices)

            traces = pca_transform(pca, traces)
            
            for plain_idx in range(self.num_plain_texts):
                curr_trace = traces[plain_idx]

                for key_hyp in range(num_keys):
                    key_hw = HAMMING_WEIGHTS[SBOX_MAT[plain_texts[plain_idx][0] ^ key_hyp]]

                    multi_gauss = multivariate_normal(gauss_mean[key_hw], gauss_cov[key_hw])
                    prob_hyp = multi_gauss.pdf(curr_trace)

                    keys_lh[key_true, key_hyp, plain_idx:] -= np.log(prob_hyp)

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