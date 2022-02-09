import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import softmax
import torch

from utils.persistence import load_json, save_numpy
from aidenv.api.config import get_program_log_dir
from aidenv.api.basic.config import build_task_kwarg
from aidenv.api.mlearn.task import MachineLearningTask
from sca.config import build_model_object, OmegaConf
from sca.file.params import str_hex_bytes, TRACE_SIZE

class GradCamSegmentation(MachineLearningTask):
    '''
    Deep learning task which extract the GRAD-CAM frequency segmentation from a trace
    window frequency classifier.
    '''

    def __init__(self, loader, voltages, frequencies, key_values,
                    plain_bounds, training_path, checkpoint_file, batch_size,
                    interp_kind, trace_len, num_workers, workers_type):
        '''
        Create new GRAD-CAM frequency segmentation.
        loader: power trace loader
        voltages: voltages of platforms to segment
        frequencies: frequencies of platforms to segment
        plain_bounds: start, end plain text indices
        training_path: root directory of a model training
        checkpoint_file: file name of the model checkpoint
        batch_size: batch size for model inference
        interp_kind: interpolation kind for map upscaling
        trace_len: size of the trace to segment
        num_workers: number of processes to split workload
        workers_type: type of joblib workers
        '''
        self.loader = loader
        self.assembler = None
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
        if interp_kind is None:
            self.interp_kind = 'linear'
        else:
            self.interp_kind = interp_kind
        if trace_len is None:
            self.trace_len = TRACE_SIZE
        else:
            self.trace_len = trace_len
        self.num_workers = num_workers
        self.workers_type = workers_type

    @classmethod
    @build_task_kwarg('loader')
    def build_kwargs(cls, config):
        pass

    def segmentations_work(self, *args):
        '''
        Work method of one process computing all plains segmented traces for a given key.
        '''
        segmented_traces = np.zeros((self.num_plain_texts, self.num_classes, self.trace_len))

        traces = self.assembler.make_traces(*args)
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
            maps = self.model.module.encoder.forward_hooks['grad_cam'].detach().cpu().numpy()
            num_maps = maps.shape[1]
            maps_size = maps.shape[2]

            for j in range(real_batch_size):
                batch_maps = maps[j].T
                batch_weights = np.zeros((self.num_classes, num_maps))
                
                # grad weighted maps
                for k in range(self.num_classes):
                    pred = y_hat[j, k]
                    pred.backward(retain_graph=True)
                    grad = self.model.module.encoder.backward_hooks['grad_cam'][0].detach().cpu().numpy()
                    batch_weights[k] = grad

                class_maps = np.matmul(batch_maps, batch_weights.T)
                #class_maps[class_maps<0.] = 0.
                #class_max = np.max(class_maps, axis=0)
                #class_maps = np.nan_to_num(class_maps / class_max.T)
                class_maps = softmax(class_maps, axis=1)

                # maps upscaling
                upscaled_maps = np.zeros((self.num_classes, self.trace_len))

                for k in range(self.num_classes):
                    x_origin = np.linspace(0, self.trace_len, maps_size)
                    scaling_interp = interp1d(x_origin, class_maps[:,k], kind=self.interp_kind)
                    x_scaled = np.linspace(0, self.trace_len, self.trace_len)
                    upscaled_maps[k] = scaling_interp(x_scaled)

                upscaled_maps[upscaled_maps<0.] = 0.
                segmented_traces[i*self.batch_size + j] = upscaled_maps
        
        return segmented_traces

    def compute_segmentations(self, log_dir):
        '''
        Compute traces frequency segmentations
        '''
        raise NotImplementedError

    def run(self, *args):
        log_dir = get_program_log_dir()

        training_path = os.path.join(self.training_path, 'program.yaml')
        training_config = OmegaConf.load(training_path)

        lookup_path = os.path.join(training_config.dataset.params.lookup_path, 'params.json')
        classif_frequencies = load_json(lookup_path)['frequencies']
        self.classifier_frequencies = classif_frequencies
        self.num_classes = len(classif_frequencies)
        print(f'Found {len(self.classifier_frequencies)} classified frequencies')

        model = build_model_object(training_config.model)
        print('Loaded classifier model')
        labels = self.classifier_frequencies
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

        self.compute_segmentations(log_dir)