from scipy.interpolate import interp1d

class FrequencyRescaler:
    '''
    Rescales a window into another frequency.
    '''

    def __init__(self, window_size, freq_ratio, interp_kind='linear'):
        self.window_size = window_size
        self.freq_ratio = freq_ratio
        self.interp_kind = interp_kind

    def scale_window(self, window):
        time = np.arange(0, window_size)
        f_interp = interp1d(time, window, kind=self.interp_kind)
        time_interp = time * self.freq_ratio
        window_interp = f_interp(time_interp, window)
        return window_interp