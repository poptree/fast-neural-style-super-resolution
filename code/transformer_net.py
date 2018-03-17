import torch
import torch.nn as nn
import numpy as np
"""
class ConvLayer(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class InstanceNormalization(torch.nn.Module):
    
    def __init__(self, dim, eps = 1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        # n = h * w
        n = x.size(2)*x.size(3)
        # reshape x from (N, CH, H, W) to (N, CH, H * W)
        t = x.view(x.size(0), x.size(1), n)
        # calculate the mean number of the (N, CH) 
        # mean = (N, CH, 1, 1, (H * W))
        # the meddle 1 are used to protect (H * W) without division
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))

        # seems like munaul broadcast?
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)

        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out

class ResidualBlock(torch.nn.Module):
    
    def __init__(self, channels, use_instance_norm = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size = 1, stride = 1)
        if use_instance_norm:
            self.in1 = InstanceNormalization(channels)
        else:
            self.in1 = nn.BatchNorm2d(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size = 3, stride = 1)
        if use_instance_norm:
            self.in2 = InstanceNormalization(channels)
        else:
            self.in2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsamleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample = None):
        super(UpsamleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor = upsample)
        
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels,kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
    
    
class TransformerNet(torch.nn.Module):
    def __init__(self, opts):
        super(TransformerNet, self).__init__()
        prev_dim = 3
        self.layers = []
        opts = opts.split(',')
        print(opts)
        for i, opt in enumerate(opts):
            print(i,"----",opt)
            next_dim = prev_dim
            needs_relu = True
            needs_bn = True
            # conv layer
            if opt[0] == 'c':
                c = int(opt[1:2])
                s = int(opt[3:4])
                next_dim = int(opt[5:])
                self.layers.append(ConvLayer(prev_dim, next_dim, kernel_size = c, stride = s))
            elif opt[0] == 'd':
                next_dim = int(opt[1:])
                self.layers.append(ConvLayer(prev_dim, next_dim, kernel_size = 3, stride = 2))
            elif opt[0] == 'R':
                next_dim = int(opt[1:])
                self.layers.append(ResidualBlock(next_dim))
                needs_bn = False
                needs_relu = False
            elif opt[0] == 'r':
                next_dim = int(opt[1:])
                self.layers.append(ResidualBlock(next_dim, use_instance_norm = False))
                needs_bn = False
                needs_relu = False
            elif opt[0] == 'u':
                next_dim = int(opt[1:])
                self.layers.append(UpsamleConvLayer(prev_dim, next_dim, kernel_size = 3, stride = 1, upsample = 2))
            elif opt[0] == 'U':
                next_dim = prev_dim
                factor = int(opt[1:])
                self.layers.append(nn.Upsample(scale_factor = factor))
            if needs_bn:
                self.layers.append(InstanceNormalization(next_dim))
            if needs_relu:
                self.layers.append(nn.ReLU())
            prev_dim = next_dim
        for idd,m in enumerate(self.layers):
            self.add_module('l'+str(idd),m)
        #print(self.layers)
        #return nn.Sequential(*self.layer)

    def forward(self, x):
        in_x = x.clone()
        y = in_x.clone()
        for name,m in enumerate(self.modules()):
            if name != 0:
                y = m(y)
        #y=self.modules()(y)
        return y

"""
"""
class TransformerNet(torch.nn.Module):
    def __init__(self, opts):
        super(TransformerNet, self).__init__()
        prev_dim = 3
        self.layers = []
        opts = opts.split(',')
        print(opts)
        for i, opt in enumerate(opts):
            next_dim = prev_dim
            needs_relu = True
            needs_bn = True
            # conv layer
            if opt[0] == 'c':
                c = int(opt[1:2])
                s = int(opt[3:4])
                next_dim = int(opt[5:])
                self.layers.append(ConvLayer(prev_dim, next_dim, kernel_size = c, stride = s))
            elif opt[0] == 'd':
                next_dim = int(opt[1:])
                self.layers.append(ConvLayer(prev_dim, next_dim, kernel_size = 3, stride = 2))
            elif opt[0] == 'R':
                next_dim = int(opt[1:])
                self.layers.append(ResidualBlock(next_dim))
                needs_bn = False
                needs_relu = False
            elif opt[0] == 'r':
                next_dim = int(opt[1:])
                self.layers.append(ResidualBlock(next_dim, use_instance_norm = False))
                needs_bn = False
                needs_relu = False
            elif opt[0] == 'u':
                next_dim = int(opt[1:])
                self.layers.append(UpsamleConvLayer(prev_dim, next_dim, kernel_size = 3, stride = 1, upsample = 2))
            elif opt[0] == 'U':
                next_dim = prev_dim
                factor = int(opt[1:])
                self.layers.append(nn.Upsample(scale_factor = factor))
            if needs_bn:
                self.layers.append(InstanceNormalization(next_dim))
            if needs_relu:
                self.layers.append(nn.ReLU())
            prev_dim = next_dim
        print(self.layers)
        ##return nn.Sequential(*self.layer)

    def forward(self, x):
        in_x = x.clone()
        y = in_x.clone()
        for l in self.layers:
            y = l(y)
        return y
"""
       


class TransformerNet(torch.nn.Module):
    def __init__(self,opts):
        super(TransformerNet, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 64, kernel_size=9, stride=1)
        self.in1 = InstanceNormalization(64)
        #self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        #self.in2 = InstanceNormalization(64)
        #self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        #self.in3 = InstanceNormalization(128)

        # Residual layers
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(32)
        #self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        #self.in5 = InstanceNormalization(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        in_X = X
        y = self.relu(self.in1(self.conv1(in_X)))
        #y = self.relu(self.in2(self.conv2(y)))
        #y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        #y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out