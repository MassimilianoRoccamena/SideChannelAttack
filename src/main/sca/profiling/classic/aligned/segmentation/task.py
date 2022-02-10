import os
import tqdm
from math import ceil
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import softmax
from scipy.ndimage import label
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
                    interp_kind, trace_len, log_segmentation, log_localization,
                    num_workers, workers_type):
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
        log_segmentation: wheter to persist segmentation results
        log_localization: wheter to persist localization results
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
        if log_segmentation is None:
            self.log_segmentation = False
        else:
            self.log_segmentation = log_segmentation
        if log_localization is None:
            self.log_localization = True
        else:
            self.log_localization = log_localization
        self.num_workers = num_workers
        self.workers_type = workers_type
        if not self.num_workers is None:
            raise NotImplementedError('multiprocessing WIP')
        self.log_dir = get_program_log_dir()

    @classmethod
    @build_task_kwarg('loader')
    def build_kwargs(cls, config):
        pass

    def segmentation_work(self, *args):
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

    def localization_work(self, key_value, segmented_traces):
        '''
        Work method of one process computing all plains windows localization for a given key.
        '''
        df_windows = pd.DataFrame(columns=['plain_index','time_start','time_end','frequency'])

        for plain_idx in range(self.num_plain_texts):
            plain_segm = np.argmax(segmented_traces[plain_idx], axis=0)
            time_start = 0

            for freq_idx in range(self.num_classes):
                class_segm = plain_segm==freq_idx
                if np.count_nonzero(class_segm) == 0:
                    continue

                class_segm = class_segm.astype(int)
                win_lclz, num_win = label(class_segm)

                for win_id in range(1,num_win+1):
                    win_idx = np.where(win_lclz==win_id)[0]
                    time_start = win_idx[0]
                    time_end = win_idx[-1]

                    df_windows = df_windows.append({'plain_index':plain_idx,'time_start':time_start, \
                                        'time_end':time_end+1, 'frequency':self.frequencies[freq_idx]}, \
                                        ignore_index=True)
            
        return df_windows

    def compute_work(self):
        '''
        Segment traces by frequencies, then localize static frequency windows.
        '''
        raise NotImplementedError

    def run(self, *args):
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

        self.compute_work()