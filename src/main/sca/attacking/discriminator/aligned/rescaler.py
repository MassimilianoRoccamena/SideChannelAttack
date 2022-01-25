import numpy as np
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
        windows_interp = f_interp(time_interp)
        return windows_interp
    
    #def scale_windows(self, windows):
    #    '''
    #    Rescale a window/trace to another frequency using interpolation.
    #    '''
    #    windows_size = windows.shape[-1]

    #    time1 = np.arange(0, windows_size)
    #    f1 = interp1d(time1, windows, kind=self.interp_kind)

    #    time2 = np.linspace(0, windows_size-1, num=int(windows_size/self.freq_ratio))
    #    windows2 = f1(time2)
    #    f2 = interp1d(time2, windows2, kind=self.interp_kind)

    #    windows3 = f2(time1)
    #    return windows3

    #def scale_windows(self, windows):
    #    windows_size = windows.shape[-1]
    #    time = np.arange(0, windows_size)
    #    f_interp = interp1d(time, windows, kind=self.interp_kind)
    #    time_interp = np.linspace(0, windows_size-1, num=int(windows_size/self.freq_ratio))
    #    windows_interp = f_interp(time_interp)
    #    return windows_interp