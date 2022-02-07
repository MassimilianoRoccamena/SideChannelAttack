import torch
import torch.nn as nn
import torch.nn.functional as F

from aidenv.api.dlearn.module.encoder import EncoderModule
from aidenv.api.dlearn.module.conv import Conv1dPadSame, MaxPool1dPadSame

class VggLayer(nn.Module):
    '''
    VGG bn + n_convs + pool layer.
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_convs, use_batch_norm):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(in_channels)

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(num_convs):
            conv = Conv1dPadSame(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride)
            self.convs.append(conv)
            relu = nn.ReLU()
            self.relus.append(relu)
            in_channels = out_channels

    def forward(self, x):
        out = x
        if self.use_batch_norm:
            out = self.bn(x)
        for conv,relu in zip(self.convs,self.relus):
            out = conv(out)
            out = relu(out)

        return out

class VGG(EncoderModule):
    '''
    VGG encoder
    '''

    def __init__(self, encoding_size, base_filters, conv_kernel_size, conv_stride, pool_kernel_size,
                        in_channels=1, use_batch_norm=True, num_convs1=2, num_convs2=3,
                        num_layers1=3, num_layers2=2, gain_filters=2, use_final_do=True, final_do_val=0.5):
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.pool_kernel_size = pool_kernel_size
        self.use_batch_norm = use_batch_norm
        self.base_filters = base_filters
        self.num_convs1 = num_convs1
        self.num_convs2 = num_convs2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.gain_filters = gain_filters

        super().__init__(encoding_size, use_final_do, final_do_val=final_do_val)

        # layers1
        self.layers1 = nn.ModuleList()
        self.poolings1 = nn.ModuleList()
        out_channels = base_filters
        for i in range(num_layers1):
            if i == 0:
                ubn = False
            else:
                ubn = use_batch_norm
            layer = VggLayer(in_channels, out_channels, conv_kernel_size,
                                conv_stride, num_convs1, ubn)
            self.layers1.append(layer)
            self.poolings1.append(MaxPool1dPadSame(kernel_size=pool_kernel_size))
            in_channels = out_channels
            out_channels *= gain_filters

        # layers2
        self.layers2 = nn.ModuleList()
        self.poolings2 = nn.ModuleList()
        for i in range(num_layers2):
            layer = VggLayer(in_channels, out_channels, conv_kernel_size,
                                conv_stride, num_convs2, use_batch_norm)
            self.layers2.append(layer)
            if i != num_layers2-1:
                self.poolings2.append(MaxPool1dPadSame(kernel_size=pool_kernel_size))
            in_channels = out_channels
            out_channels *= gain_filters

        # final
        self.set_final_size(int(out_channels / gain_filters))

    def forward(self, x):
        out = x

        for layer,pooling in zip(self.layers1, self.poolings1):
            out = layer(out)
            out = pooling(out)
        for i,layer in enumerate(self.layers2):
            out = layer(out)
            if i != self.num_layers2-1:
                out = self.poolings2[i](out)

        out.register_hook(self.forward_grad_cam_hook)
        out.register_hook(self.backward_grad_cam_hook)

        out = out.mean(-1)
        out = super().forward(out)
        return out