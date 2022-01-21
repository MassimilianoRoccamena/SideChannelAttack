from scipy.interpolate import interp1d

class FrequencyRescaler:
    '''
    Rescales a window into another frequency.
    '''

    def __init__(self, freq_ratio, interp_kind='linear'):
        self.freq_ratio = freq_ratio
        self.interp_kind = interp_kind

    def scale_windows(self, windows):
        '''
        Rescale a window/trace to another frequency using interpolation.
        '''
        windows_size = windows.shape[-1]
        time = np.arange(0, windows_size)
        f_interp = interp1d(time, windows, kind=self.interp_kind)
        time_interp = time * self.freq_ratio
        windows_interp = f_interp(time_interp, windows)
        return windows_interp