import os
import tqdm
from math import ceil
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from utils.persistence import load_pickle, load_numpy, save_numpy
from utils.math import BYTE_SIZE, BYTE_HW_LEN, pca_transform
from aidenv.api.config import get_program_log_dir
from aidenv.api.mlearn.task import MachineLearningTask
from sca.config import OmegaConf, build_task_object
from sca.assembler import DynamicAssemblerLoader, DynamicAssemblerParallelLoader
from sca.file.params import TRACE_SIZE, str_hex_bytes, SBOX_MAT, HAMMING_WEIGHTS
from sca.rescaler import FrequencyRescaler
from sca.attacking.loader import *
from sca.attacking.classic.task import StaticDiscriminator

class AlignedStaticDiscriminator(StaticDiscriminator):
    '''
    Trace key discriminator on aligned static frequency traces.
    '''

    def __init__(self, generator_path, voltages, frequencies, key_values,
                        plain_bounds, target_volt, target_freq,
                        interp_kind=None, num_workers=None, workers_type= None):
        '''
        Create new template attacker on aligned static traces.
        generator_path: path of a trace generator
        voltages: voltages of platform to attack
        frequencies: frequencies of platform to attack
        key_values: key values to attack
        plain_bounds: start, end plain text indices
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        target_freq: voltage of the aligned platform
        target_freq: frequency of the aligned platform
        interp_kind: kind of 1D interpolation for trace rescaling
        '''
        super().__init__(generator_path, voltages, frequencies, key_values,
                            plain_bounds, num_workers, workers_type)
        self.target_volt = target_volt
        self.target_freq = target_freq
        if interp_kind is None:
            interp_kind = 'linear'
        self.interp_kind = interp_kind

    def process_traces(self, voltage, frequency, key_value, traces):
        freq_ratio = float(self.target_freq) / float(frequency)
        rescaler = FrequencyRescaler(freq_ratio, self.interp_kind)
        return rescaler.scale_windows(traces)

    def target_platform(self, voltage, frequency):
        return (self.target_volt, self.target_freq)

class AlignedDynamicDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some dynamic
    frequency traces using a fitted reduced trace generative model with a frequency
    aligner model.
    '''

    def __init__(self, dynamic_path, generator_path, localization_path,
                        key_values, plain_bounds, target_freq, skip_size_lim,
                        interp_kind=None, num_workers=None, workers_type=None):
        '''
        Create new template attacker on aligned dynamic traces.
        dynamic_path: path of dynamic traces lookup data
        generator_path: path of a trace generator
        localization_path: path of frequency localization of traces
        key_values: key values to attack
        plain_bounds: start, end plain text indices
        target_freq: frequency of the aligned platform
        skip_size_lim: max size of a window to be considered valid for rescaling
        interp_kind: kind of 1D interpolation for window rescaling
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.dynamic_path = dynamic_path
        dynamic_path = os.path.join(dynamic_path, 'program.yaml')
        self.dynamic_config = OmegaConf.to_object(OmegaConf.load(dynamic_path))
        loader_config = self.dynamic_config['core']['params']['loader']['params']
        loader = build_task_object(self.dynamic_config['core']['params']['loader'])
        self.training_path = self.dynamic_config['core']['params']['training_path']
        training_path = os.path.join(self.training_path, 'program.yaml')
        self.training_config = OmegaConf.to_object(OmegaConf.load(training_path))
        self.window_path = self.training_config['dataset']['params']['window_path']
        window_path = os.path.join(self.window_path, 'program.yaml')
        self.window_config = OmegaConf.to_object(OmegaConf.load(window_path))
        self.voltages = self.window_config['core']['params']['voltages']
        self.target_volt = self.voltages[0]
        print(f'Found {len(self.voltages)} voltages')
        dynamic_path = os.path.join(self.dynamic_path, 'assembler')
        self.frequencies = self.window_config['core']['params']['frequencies']
        print(f'Found {len(self.frequencies)} frequencies')
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        elif type(key_values) is int:
            key_values = str_hex_bytes()[:key_values]
            print(f'Using first {len(key_values)} byte values')
        self.key_values = key_values
        print('Using all key values')
        self.generator_path = generator_path
        generator_path = os.path.join(generator_path, 'program.yaml')
        self.generator_config = OmegaConf.to_object(OmegaConf.load(generator_path))
        self.reduced_dim = self.generator_config['core']['params']['reduced_dim']
        self.localization_path = localization_path
        self.plain_bounds = plain_bounds
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.target_freq = target_freq
        self.skip_size_lim = skip_size_lim
        if interp_kind is None:
            interp_kind = 'linear'
        self.interp_kind = interp_kind
        self.num_workers = num_workers
        self.workers_type = workers_type
        if self.num_workers is None:
            self.assemb_loader = DynamicAssemblerLoader(loader, self.target_volt, dynamic_path)
            self.parallel_mode = False
        else:
            print('Running in parallel mode')
            self.assemb_loader = DynamicAssemblerParallelLoader(loader, self.target_volt, dynamic_path, \
                                                                num_workers, workers_type)
            self.parallel_mode = True
        self.log_dir = get_program_log_dir()

    def align_trace(self, trace, key_true, plain_index):
        '''
        Align a trace to target frequency using localization output and frequency rescaling.
        '''
        df_plain = self.df_true[self.df_true['plain_index']==plain_index]
        df_plain.sort_values(by='time_start', inplace=True)
        time_indices = df_plain[['time_start','time_end']].to_numpy()
        frequencies = df_plain['frequency'].to_numpy()
        n_switches = frequencies.shape[0]
        aligned_trace = np.zeros(TRACE_SIZE)

        start_idx = 0

        for i in range(n_switches):
            curr_idx = time_indices[i]
            curr_start = curr_idx[0]
            curr_end = curr_idx[1]
            curr_size = curr_end - curr_start
            curr_freq = frequencies[i]

            if curr_freq == self.target_freq:
                end_idx = start_idx + curr_size
                aligned_trace[start_idx:end_idx] = trace[curr_start:curr_end]
                start_idx += curr_size
            elif curr_end - curr_start < self.skip_size_lim:
                print(f'encountered window with size {curr_end-curr_start} for key {key_true} and plain index {plain_index}')
                end_idx = start_idx + curr_size
                aligned_trace[start_idx:end_idx] = trace[curr_start:curr_end]
                start_idx += curr_size
            else:
                freq_ratio = float(self.target_freq) / float(curr_freq)
                rescaler = FrequencyRescaler(freq_ratio, self.interp_kind)
                aligned_win = rescaler.scale_windows(trace[curr_start:curr_end])
                size_aligned = aligned_win.shape[-1]
                end_idx = start_idx + size_aligned
                aligned_trace[start_idx:end_idx] = aligned_win
                start_idx += size_aligned

        trace = aligned_trace[:trace.shape[-1]]
        return trace

    def key_likelihood_work(self, pca, gauss_mean, gauss_cov, key_true):
        '''
        Work method of one process computing one true key likelihoods over all key hypothesis.
        '''
        num_keys = len(self.key_values)
        key_lh = np.zeros((num_keys, self.num_plain_texts))

        lclz_path = os.path.join(self.localization_path, f'{key_true}.csv')
        self.df_true = pd.read_csv(lclz_path)

        if not self.parallel_mode:
            traces = np.zeros((self.num_plain_texts, self.assemb_loader.loader.trace_len))
            plain_texts = np.zeros((self.num_plain_texts, 16), dtype='uint8')
            for plain_idx in range(self.num_plain_texts):
                trace, plain_text, key = self.assemb_loader.fetch_trace(key_true, plain_idx)
                traces[plain_idx] = trace
                plain_texts[plain_idx] = plain_text
                
        else:
            traces = np.zeros((self.num_plain_texts, self.assemb_loader.loader.trace_len))
            plain_texts = np.zeros((self.num_plain_texts, 16), dtype='uint8')
            plain_indices = self.plain_indices - self.plain_indices[0]
            wdata = self.assemb_loader.fetch_traces(key_true, plain_indices)
            for curr_data in wdata:
                plain_indices, ftraces, fplains, key = curr_data
                traces[plain_indices] = ftraces
                plain_texts[plain_indices] = fplains
            
        for plain_idx in range(self.num_plain_texts):
            trace = traces[plain_idx]
            plain_text = plain_texts[plain_idx]
            trace = self.align_trace(trace, key_true, plain_idx)
            trace = np.reshape(trace, (1,*trace.shape))
            trace = pca_transform(pca, trace)[0]

            for key_hyp in range(num_keys):
                sbox_hw = HAMMING_WEIGHTS[SBOX_MAT[plain_text[0] ^ key_hyp]]

                multi_gauss = multivariate_normal(gauss_mean[sbox_hw], gauss_cov[sbox_hw])
                prob_hyp = multi_gauss.pdf(trace)
                    
                key_lh[key_hyp, plain_idx] -= np.log(prob_hyp)

        file_path = os.path.join(self.lh_path, f'{key_true}.npy')
        save_numpy(key_lh, file_path)

    def compute_work(self):
        '''
        Compute likelihoods of keys for the dynamic traces.
        '''
        template_path = os.path.join(self.generator_path, f'{self.target_volt}-{self.target_freq}')
        curr_path = os.path.join(template_path, 'pca')
        pca = load_pickle(os.path.join(curr_path, 'pca.pckl'))
        curr_path = os.path.join(template_path, 'multi_gauss')
        gauss_mean = load_numpy(os.path.join(curr_path, 'mean.npy'))
        gauss_cov = load_numpy(os.path.join(curr_path, 'covariance.npy'))
        num_keys = len(self.key_values)

        pbar = tqdm.tqdm(total=num_keys)
        for key_true in self.key_values:
            self.key_likelihood_work(pca, gauss_mean, gauss_cov, key_true)
            pbar.update(1)

        pbar.close()

    def run(self, *args):
        self.lh_path = os.path.join(self.log_dir, 'likelihood')
        os.mkdir(self.lh_path)

        print('Computing keys likelihood\n')
        self.compute_work()
