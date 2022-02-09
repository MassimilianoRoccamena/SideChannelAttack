import os
import tqdm
import math
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from utils.persistence import save_pickle, save_json, save_numpy
from utils.math import BYTE_HW_LEN, kahan_sum, pca_transform
from aidenv.api.config import get_program_log_dir
from aidenv.api.basic.config import build_task_kwarg
from aidenv.api.mlearn.task import MachineLearningTask
from sca.file.params import TRACE_SIZE, str_hex_bytes, SBOX_MAT, HAMMING_WEIGHTS
from sca.loader import *

class TraceGenerator(MachineLearningTask):
    '''
    Machine learning task which fits the generative distribution of a dimensional reduced power trace
    given the hamming weight of byte0 of the key.
    '''

    def __init__(self, loader, voltages, frequencies, key_values, plain_bounds, reduced_dim):
        '''
        Create new PCA+QDA template attacker.
        loader: power trace loader
        voltages: device voltages
        frequencies: device frequencies
        key_values: key values of the encryption
        plain_bounds: start, end plain text indices
        reduced_dim: dimensionality reduction size
        log_dir: log directory of the task
        '''
        self.loader = loader
        self.voltages = list(voltages)
        self.frequencies = list(frequencies)
        if key_values is None:
            key_values = str_hex_bytes()
        self.key_values = list(key_values)
        self.hw_len = BYTE_HW_LEN
        self.plain_bounds = list(plain_bounds)
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.reduced_dim = reduced_dim

    @classmethod
    @build_task_kwarg('loader')
    def build_kwargs(cls, config):
        pass

    def fit_pca(self, voltage, frequency):
        '''
        Fit a PCA dimensionality reducer of trace data given the (voltage,frequency) platform.
        '''
        D = np.zeros(self.hw_len, dtype='int')
        sum_hw = np.zeros((self.hw_len, TRACE_SIZE), dtype='float64')
        c_hw = np.zeros((self.hw_len, TRACE_SIZE), dtype='float64')
        pbar = tqdm.tqdm(total=len(self.key_values))
        
        for key_value in self.key_values:
            file_id = self.loader.build_file_id(voltage, frequency, key_value)
            file_path = self.loader.build_file_path(file_id)
            traces, plain_texts, key = self.loader.load_some_traces(file_path, self.plain_indices)
                
            s = SBOX_MAT[plain_texts[:,0] ^ key[0]]
            h = HAMMING_WEIGHTS[s]

            for j in range(self.hw_len):
                sum_hw[j], c_hw[j]= kahan_sum(sum_hw[j], c_hw[j], np.sum(traces[h==j], 0))
                D[j] += np.sum(h==j)

            pbar.update(1)
            
        pbar.close()
        mean = sum_hw / D.reshape((self.hw_len,1)).repeat(TRACE_SIZE, axis=1)
            
        pca = PCA(self.reduced_dim)
        mean_reduced = pca.fit_transform(mean)
        print(f'\nExplained variance is {np.sum(pca.explained_variance_ratio_)}')

        return mean, pca, mean_reduced

    def fit_multi_gauss(self, voltage, frequency, pca):
        '''
        Fit the generative distribution of PCA reduced traces given the (voltage,frequency) platform.
        '''
        t_HW = [[] for i in range(self.hw_len)]
        pbar = tqdm.tqdm(total=len(self.key_values))

        for key_value in self.key_values:
            file_id = self.loader.build_file_id(voltage, frequency, key_value)
            file_path = self.loader.build_file_path(file_id)
            traces, plain_texts, key = self.loader.load_some_traces(file_path, self.plain_indices)

            traces = pca_transform(pca, traces)
            
            s = SBOX_MAT[plain_texts[:,0] ^ key[0]]
            h = HAMMING_WEIGHTS[s]
                
            for j in range(self.hw_len):
                t_HW[j].extend(traces[h==j])

            pbar.update(1)
            
        pbar.close()
        t_HW = [np.array(t_HW[h], dtype=traces.dtype) for h in range(self.hw_len)]
        
        gauss_mean  = np.zeros((self.hw_len, self.reduced_dim))
        gauss_cov  = np.zeros((self.hw_len, self.reduced_dim, self.reduced_dim))
        for HW in range(self.hw_len):
            gauss_mean[HW]= np.mean(t_HW[HW], 0)
            for i in range(self.reduced_dim):
                for j in range(self.reduced_dim):
                    x = t_HW[HW][:,i]
                    y = t_HW[HW][:,j]
                    gauss_cov[HW,i,j] = np.cov(x, y)[0][1]

        return gauss_mean, gauss_cov

    def run(self, *args):
        log_dir = get_program_log_dir()

        # init params
        params = {'voltages':self.voltages,'frequencies':self.frequencies,
                    'key_values':self.key_values,'plain_bounds':self.plain_bounds,
                    'reduced_dim':self.reduced_dim}

        # platforms
        for voltage in self.voltages:
            for frequency in self.frequencies:
                print(f'\nProcessing {voltage}-{frequency} platform')
                curr_root_path = os.path.join(log_dir, f'{voltage}-{frequency}')
                if not os.path.exists(curr_root_path):
                    os.mkdir(curr_root_path)

                # pca
                print('Reducing data dimensionality\n')
                curr_path = os.path.join(curr_root_path, 'pca')
                if not os.path.exists(curr_path):
                    os.mkdir(curr_path)

                mean, pca, mean_reduced = self.fit_pca(voltage, frequency)
                save_numpy(mean, os.path.join(curr_path, 'mean.npy'))
                save_pickle(pca, os.path.join(curr_path, 'pca.pckl'))
                save_numpy(mean_reduced, os.path.join(curr_path, 'mean_reduced.npy'))

                # multivariate gaussian
                print('\nEstimating generative distribution\n')
                curr_path = os.path.join(curr_root_path, 'multi_gauss')
                if not os.path.exists(curr_path):
                    os.mkdir(curr_path)

                gauss_mean, gauss_cov = self.fit_multi_gauss(voltage, frequency, pca)
                save_numpy(gauss_mean, os.path.join(curr_path, 'mean.npy'))
                save_numpy(gauss_cov, os.path.join(curr_path, 'covariance.npy'))

        # save params
        params_path = os.path.join(log_dir, 'params.json')
        save_json(params, params_path)