import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
import torch

from utils.persistence import save_numpy
from utils.math import BYTE_SIZE, BYTE_HW_LEN
from aidenv.api.config import get_program_log_dir
from aidenv.api.mlearn.task import MachineLearningTask
from sca.config import OmegaConf, build_task_object, build_model_object
from sca.assembler import DynamicAssemblerLoader, DynamicAssemblerParallelLoader
from sca.file.params import SBOX_MAT, HAMMING_WEIGHTS
from sca.file.params import str_hex_bytes
from sca.file.loader import ParallelTraceLoader
from sca.attacking.loader import *

class DeepStaticDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some static
    frequency traces using a fitted deep classifier.
    '''

    def __init__(self, training_path, checkpoint_file, voltages, frequencies, key_values,
                        plain_bounds, batch_size, num_workers=None, workers_type=None):
        '''
        Create new static deep key attacker.
        training_path: root directory of a trained sbox hw classifier
        checkpoint_file: file name of the model checkpoint
        voltages: voltages of platforms to attack
        frequencies: frequencies of platforms to attack
        key_values: key values to attack
        plain_bounds: start, end plain text indices
        batch_size: batch size for model inference
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.training_path = training_path
        self.checkpoint_file = checkpoint_file
        training_path = os.path.join(training_path, 'program.yaml')
        self.training_config = OmegaConf.to_object(OmegaConf.load(training_path))
        loader = build_task_object(self.training_config['dataset']['params']['loader'])
        self.model = None
        if voltages is None:
            self.voltages = self.training_config['dataset']['params']['voltages']
            print(f'Found {len(self.voltages)} voltages')
        else:
            self.voltages = voltages
        if frequencies is None:
            self.frequencies = self.training_config['dataset']['params']['frequencies']
            print(f'Found {len(self.frequencies)} frequencies')
        else:
            self.frequencies = frequencies
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        elif type(key_values) is int:
            key_values = str_hex_bytes()[:key_values]
            print(f'Using first {len(key_values)} byte values')
        self.key_values = key_values
        self.plain_bounds = plain_bounds
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running computations on {self.device}')
        self.num_workers = num_workers
        self.workers_type = workers_type
        if self.num_workers is None:
            self.loader = loader
            self.parallel_mode = False
        else:
            print('Running in parallel mode')
            self.loader = ParallelTraceLoader(loader, num_workers, workers_type)
            self.parallel_mode = True
        self.log_dir = get_program_log_dir()

    def key_likelihood_work(self, voltage, frequency, key_true):
        '''
        Work method of one process computing one true key likelihoods over all key hypothesis.
        '''
        num_keys = len(self.key_values)
        key_lh = np.zeros((num_keys, self.num_plain_texts))
        
        for key_hyp in range(num_keys):
            n_iters = ceil(self.num_plain_texts / self.batch_size)

            for i in range(n_iters):
                low_plain_idx = i*self.batch_size
                high_plain_idx = min((i+1)*self.batch_size-1, self.num_plain_texts-1)
                real_batch_size = high_plain_idx - low_plain_idx + 1

                if not self.parallel_mode:
                    plain_idx = i*self.batch_size + j
                    file_id = self.loader.build_file_id(voltage, frequency, key_true)
                    file_path = self.loader.build_file_path(file_id)
                    plain_indices = self.plain_indices[low_plain_idx:high_plain_idx+1]
                    traces, plain_texts, key = self.loader.fetch_traces(file_path, plain_indices)
                else:
                    traces = np.zeros((real_batch_size, self.loader.loader.trace_len), dtype='float32')
                    plain_texts = np.zeros((real_batch_size, 16), dtype='uint8')
                    plain_indices = [i for i in range(low_plain_idx, high_plain_idx+1)]

                    wdata = self.loader.fetch_traces(voltage, frequency, key_true, plain_indices)
                    for curr_data in wdata:
                        plain_idx, ftraces, fplains, key = curr_data
                        plain_idx = np.array(plain_idx) - low_plain_idx
                        traces[plain_idx] = ftraces
                        plain_texts[plain_idx] = fplains

                traces = traces.reshape(real_batch_size, 1, traces.shape[-1])
                traces = torch.from_numpy(traces)
                traces = traces.to(self.device)
                y_hat = self.model(traces)

                for j in range(real_batch_size):
                    plain_text = plain_texts[j][0]
                    sbox_hw = HAMMING_WEIGHTS[SBOX_MAT[plain_text ^ key_hyp]]
                    prob_hyp = y_hat[j, sbox_hw].detach().cpu().numpy()
                    prob_hyp = np.maximum(prob_hyp, np.ones(prob_hyp.shape)*(1e-200))
                    plain_idx = i*self.batch_size + j
                    key_lh[key_hyp, plain_idx] -= np.log(prob_hyp)

        file_path = os.path.join(self.lh_path, f'{key_true}.npy')
        save_numpy(key_lh, file_path)

    def compute_work(self, voltage, frequency):
        '''
        Compute likelihoods of keys for the attack traces of the (voltage,frequency) platform.
        '''
        num_keys = len(self.key_values)

        pbar = tqdm.tqdm(total=num_keys)
        for key_true in self.key_values:
            self.key_likelihood_work(voltage, frequency, key_true)
            pbar.update(1)

        pbar.close()

    def run(self, *args):
        model = build_model_object(self.training_config['model'])
        print('Loaded classifier model')

        labels = [str(i) for i in range(BYTE_HW_LEN)]
        model.module.set_labels(labels)
        checkpoint_path = os.path.join(self.training_path, 'checkpoints', self.checkpoint_file)
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        if 'loss.weight' in checkpoint.keys():
            del checkpoint['loss.weight']
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        self.model = model
        print('Loaded model checkpoint')

        for voltage in self.voltages:
            for frequency in self.frequencies:
                print(f'\nProcessing {voltage}-{frequency} platform')
                platform_path = os.path.join(self.log_dir, f'{voltage}-{frequency}')
                os.mkdir(platform_path)
                self.lh_path = platform_path
                print('Computing keys likelihood\n')
                self.compute_work(voltage, frequency)

class DeepDynamicDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some dynamic
    frequency traces using a fitted deep classifier.
    '''

    def __init__(self, dynamic_path, training_path, checkpoint_file, key_values,
                        plain_bounds, batch_size, num_workers=None, workers_type=None):
        '''
        Create new dynamic deep key attacker.
        dynamic_path: path of dynamic traces lookup data
        training_path: root directory of a trained sbox hw classifier
        checkpoint_file: file name of the model checkpoint
        key_values: key values to attack
        plain_bounds: start, end plain text indices
        batch_size: batch size for model inference
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.dynamic_path = dynamic_path
        dynamic_path = os.path.join(dynamic_path, 'program.yaml')
        self.dynamic_config = OmegaConf.to_object(OmegaConf.load(dynamic_path))
        loader_config = self.dynamic_config['core']['params']['loader']['params']
        loader = build_task_object(self.dynamic_config['core']['params']['loader'])
        wclassif_path = self.dynamic_config['core']['params']['training_path']
        wclassif_path = os.path.join(wclassif_path, 'program.yaml')
        wclassif_config = OmegaConf.to_object(OmegaConf.load(wclassif_path))
        self.window_path = wclassif_config['dataset']['params']['window_path']
        window_path = os.path.join(self.window_path, 'program.yaml')
        self.window_config = OmegaConf.to_object(OmegaConf.load(window_path))
        self.voltages = self.window_config['core']['params']['voltages']
        print(f'Found {len(self.voltages)} voltages')
        dynamic_path = os.path.join(self.dynamic_path, 'assembler')
        self.frequencies = self.window_config['core']['params']['frequencies']
        print(f'Found {len(self.frequencies)} frequencies')
        self.training_path = training_path
        self.checkpoint_file = checkpoint_file
        training_path = os.path.join(training_path, 'program.yaml')
        self.training_config = OmegaConf.to_object(OmegaConf.load(training_path))
        self.model = None
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        elif type(key_values) is int:
            key_values = str_hex_bytes()[:key_values]
            print(f'Using first {len(key_values)} byte values')
        self.key_values = key_values
        self.plain_bounds = plain_bounds
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running computations on {self.device}')
        self.num_workers = num_workers
        self.workers_type = workers_type
        if self.num_workers is None:
            self.assemb_loader = DynamicAssemblerLoader(loader, self.voltages[0], dynamic_path)
            self.parallel_mode = False
        else:
            print('Running in parallel mode')
            self.assemb_loader = DynamicAssemblerParallelLoader(loader, self.voltages[0], dynamic_path, \
                                                                num_workers, workers_type)
            self.parallel_mode = True
        self.log_dir = get_program_log_dir()

    def key_likelihood_work(self, key_true):
        '''
        Work method of one process computing one true key likelihoods over all key hypothesis.
        '''
        num_keys = len(self.key_values)
        key_lh = np.zeros((num_keys, self.num_plain_texts))
        
        for key_hyp in range(num_keys):
            n_iters = ceil(self.num_plain_texts / self.batch_size)

            for i in range(n_iters):
                low_plain_idx = i*self.batch_size
                high_plain_idx = min((i+1)*self.batch_size-1, self.num_plain_texts-1)
                real_batch_size = high_plain_idx - low_plain_idx + 1

                traces = np.zeros((real_batch_size, self.assemb_loader.loader.trace_len), dtype='float32')
                plain_texts = np.zeros((real_batch_size, 16), dtype='uint8')

                if not self.parallel_mode:
                    for j in range(real_batch_size):
                        plain_idx = i*self.batch_size + j
                        trace, plain_text, key = self.assemb_loader.fetch_trace(key_true, plain_idx)
                        traces[j] = trace
                        plain_texts[j] = plain_text
                else:
                    plain_indices = [i for i in range(low_plain_idx, high_plain_idx+1)]
                    wdata = self.assemb_loader.fetch_traces(key_true, plain_indices)
                    for curr_data in wdata:
                        plain_idx, trace, plain_text, key = curr_data
                        traces[plain_idx - low_plain_idx] = trace
                        plain_texts[plain_idx - low_plain_idx] = plain_text

                traces = traces.reshape(real_batch_size, 1, traces.shape[-1])
                traces = torch.from_numpy(traces)
                traces = traces.to(self.device)
                y_hat = self.model(traces)

                for j in range(real_batch_size):
                    plain_text = plain_texts[j][0]
                    sbox_hw = HAMMING_WEIGHTS[SBOX_MAT[plain_text ^ key_hyp]]
                    prob_hyp = y_hat[j, sbox_hw].detach().cpu().numpy()
                    prob_hyp = np.maximum(prob_hyp, np.ones(prob_hyp.shape)*(1e-200))
                    plain_idx = i*self.batch_size + j
                    key_lh[key_hyp, plain_idx] -= np.log(prob_hyp)

        file_path = os.path.join(self.lh_path, f'{key_true}.npy')
        save_numpy(key_lh, file_path)

    def compute_work(self):
        '''
        Compute likelihoods of keys for the attack traces of the (voltage,frequency) platform.
        '''
        num_keys = len(self.key_values)

        pbar = tqdm.tqdm(total=num_keys)                                        # vanilla
        for key_true in self.key_values:
            self.key_likelihood_work(key_true)
            pbar.update(1)

        pbar.close()

    def run(self, *args):
        self.lh_path = os.path.join(self.log_dir, 'likelihood')
        os.mkdir(self.lh_path)
        
        model = build_model_object(self.training_config['model'])
        print('Loaded classifier model')

        labels = [str(i) for i in range(BYTE_HW_LEN)]
        model.module.set_labels(labels)
        checkpoint_path = os.path.join(self.training_path, 'checkpoints', self.checkpoint_file)
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        if 'loss.weight' in checkpoint.keys():
            del checkpoint['loss.weight']
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        self.model = model
        print('Loaded model checkpoint')

        # platforms
        print('Computing keys likelihood\n')
        self.compute_work()