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
from sca.file.params import SBOX_MAT, HAMMING_WEIGHTS
from sca.file.params import str_hex_bytes
from sca.attacking.loader import *

class DeepStaticDiscriminator(MachineLearningTask):
    '''
    Machine learning task which compute the log likelihood of a key given some static
    frequency traces using a fitted deep classifier.
    '''

    def __init__(self, training_path, checkpoint_file, voltages, frequencies, key_values,
                        plain_bounds, batch_size, num_workers=None, workers_type=None):
        '''
        Create new deep key attacker.
        training_path: root directory of a model training
        checkpoint_file: file name of the model checkpoint
        voltages: voltages of platforms to attack
        frequencies: frequencies of platforms to attack
        plain_bounds: start, end plain text indices
        batch_size: batch size for model inference
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.training_path = training_path
        self.checkpoint_file = checkpoint_file
        training_path = os.path.join(training_path, 'program.yaml')
        self.training_config = OmegaConf.to_object(OmegaConf.load(training_path))
        self.loader = build_task_object(self.training_config['dataset']['params']['loader'])
        self.model = None
        if voltages is None:
            self.voltages = self.training_config['dataset']['params']['voltages']
            print(f'Found {len(self.voltages)} voltages')
        else:
            self.voltages = voltages
        if frequencies is None:
            self.frequencies = self.training_config['dataset']['params']['frequencies']
            print(f'Found {len(self.frequencies)} voltages')
        else:
            self.frequencies = frequencies
        if key_values is None:
            key_values = str_hex_bytes()
            print('Using all key values')
        self.key_values = key_values
        self.plain_bounds = plain_bounds
        self.plain_indices = np.arange(plain_bounds[0], plain_bounds[1])
        self.num_plain_texts = plain_bounds[1] - plain_bounds[0]
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers
        self.workers_type = workers_type
        if not self.num_workers is None:
            raise NotImplementedError('multiprocessing WIP')
        self.log_dir = get_program_log_dir()

    def key_likelihood_work(self, voltage, frequency, key_true):
        '''
        Work method of one process computing one true key likelihoods over all key hypothesis.
        '''
        num_keys = len(self.key_values)
        key_lh = np.zeros((num_keys, self.num_plain_texts))

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
                    key_lh[key_hyp, plain_idx:] -= np.log(prob_hyp)

        return key_lh

    def compute_work(self, voltage, frequency):
        '''
        Compute likelihoods of keys for the attack traces of the (voltage,frequency) platform.
        '''
        num_keys = len(self.key_values)

        num_workers = self.num_workers
        workers_type = self.workers_type

        if num_workers is None:
            pbar = tqdm.tqdm(total=num_keys)                                        # vanilla
            for key_true in self.key_values:
                key_lh = self.key_likelihood_work(voltage, frequency, key_true)
                file_path = os.path.join(self.lh_path, f'{key_true}.npy')
                save_numpy(key_lh, file_path)
                pbar.update(1)
        else:
            n_iters = ceil(len(self.key_values) / num_workers)                      # multiprocessed -- WIP
            pbar = tqdm.tqdm(total=n_iters)
            for i in range(n_iters):
                keys_true = self.key_values[i : min((i+1)*num_workers, num_keys-1)]
                Parallel(n_jobs=num_workers, prefer=workers_type) (delayed(self.key_likelihood_work) \
                        (voltage, frequency, key_true, keys_lh) for key_true in keys_true)
                pbar.update(1)

        pbar.close()

    def run(self, *args):
        training_path = os.path.join(self.training_path, 'program.yaml')
        training_config = OmegaConf.load(training_path)
        model = build_model_object(training_config.model)
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
        for voltage in self.voltages:
            for frequency in self.frequencies:
                print(f'\nProcessing {voltage}-{frequency} platform')
                platform_path = os.path.join(self.log_dir, f'{voltage}-{frequency}')
                os.mkdir(self.platform_path)
                self.lh_path = os.path.join(platform_path, 'likelihood')
                os.mkdir(self.lh_path)

                # likelihoods
                print('Computing keys likelihood\n')
                self.compute_work(voltage, frequency)
