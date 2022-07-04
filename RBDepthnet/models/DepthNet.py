import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
##########
from torch.autograd import Variable
from models.EdgeExtract import *

class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_EncoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class _InceptionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), width=1, drop_rate=0, use_bias=False):
        super(_InceptionBlock, self).__init__()

        self.width = width
        self.drop_rate = drop_rate

        for i in range(width):     # 0, 1, 2
            layer = nn.Sequential(
                nn.ReflectionPad2d(i*2+1),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=0, dilation=i*2+1, bias=use_bias)
                )
            setattr(self, 'layer'+str(i), layer)

        self.norm1 = norm_layer(output_nc*width)
        self.norm2 = norm_layer(output_nc)
        self.nonlinearity = nonlinearity
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc*width, output_nc, kernel_size=3, padding=0, bias=use_bias)
            )
    def forward(self, x):
        result = []

        for i in range(self.width):
            layer = getattr(self, 'layer'+str(i))
            result.append(layer(x))
        output = torch.cat(result, 1)
        output = self.nonlinearity(self.norm1(output))
        output = self.norm2(self.branch1x1(output))
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)

        return self.nonlinearity(output+x)

class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable((torch.randn(x.size()).cuda(x.data.get_device()) - 0.5) / 10.0)
        return x+noise

class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(_OutputBlock, self).__init__()
        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
                nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
                nn.Tanh()
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found'% norm_type)
    return norm_layer

class DepthNet(nn.Module):

    def __init__(self, input_nc=3, output_nc=1, ngf=64, layers=4, norm='batch', drop_rate=0, add_noise=False, weight=0.1):
        super(DepthNet, self).__init__()
        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type='PReLU')

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(3),
                                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                    norm_layer(ngf),
                                    nonlinearity)
        self.conv2 = _EncoderBlock(ngf, ngf * 2, ngf * 2, norm_layer, nonlinearity, use_bias)  # 64/128/128
        # self.conv2 = _EncoderBlock(ngf * 2, ngf * 2, ngf * 2, norm_layer, nonlinearity, use_bias)  # 64/128/128
        self.conv3 = _EncoderBlock(ngf * 2, ngf * 4, ngf * 4, norm_layer, nonlinearity, use_bias)  # 128/256/256
        self.conv4 = _EncoderBlock(ngf * 4, ngf * 8, ngf * 8, norm_layer, nonlinearity, use_bias)  # 256/512/512

        self.convRB = RB(ngf * 2, ngf)
        for i in range(layers - 4):
            conv = _EncoderBlock(ngf * 8, ngf * 8, ngf * 8, norm_layer, nonlinearity, use_bias)
            setattr(self, 'down' + str(i), conv.model)
            print('down' + str(i))

        center = []
        for i in range(7 - layers):  # 0, 1, 2
            center += [
                _InceptionBlock(ngf * 8, ngf * 8, norm_layer, nonlinearity, 7 - layers, drop_rate, use_bias)
            ]

        center += [
            _DecoderUpBlock(ngf * 8, ngf * 8, ngf * 4, norm_layer, nonlinearity, use_bias)
        ]

        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        for i in range(layers - 4):
            print('------')
            upconv = _DecoderUpBlock(ngf * (8 + 4), ngf * 8, ngf * 4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(ngf * (4 + 4), ngf * 8, ngf * 2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(ngf * (2 + 2) + output_nc, ngf * 4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(ngf * (1 + 1) + output_nc, ngf * 2, int(ngf / 2), norm_layer, nonlinearity,
                                           use_bias)

        self.output4 = _OutputBlock(ngf * (4 + 4), output_nc, 3, use_bias)
        self.output3 = _OutputBlock(ngf * (2 + 2) + output_nc, output_nc, 3, use_bias)
        self.output2 = _OutputBlock(ngf * (1 + 1) + output_nc, output_nc, 3, use_bias)
        self.output1 = _OutputBlock(int(ngf / 2) + output_nc, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, input, boundary):
        conv1 = self.pool(self.conv1(input))          # 3/64 1/2
        print('conv1-1:', conv1.shape)
        conv1 = torch.cat([conv1,boundary],1)
        print('conv1-2:', conv1.shape)
        conv1 = self.convRB(conv1)
        print('conv1-3:', conv1.shape)
        conv2 = self.pool(self.conv2.forward(conv1))  # 64/128 1/4
        print('conv2:', conv2.shape)
        conv3 = self.pool(self.conv3.forward(conv2))  # 128/256 1/8
        print('conv3:', conv3.shape)
        center_in = self.pool(self.conv4.forward(conv3))   # 256/512 1/16
        print('center_in:',center_in.shape)
        middle = [center_in]
        print('middle:',type(middle), len(middle))
        for i in range(self.layers-4):
            print('down' + str(i))
            model = getattr(self, 'down'+str(i))
            center_in = self.pool(model.forward(center_in))
            middle.append(center_in)
        center_out = self.center.forward(center_in)
        print('center_out:', center_out.shape)
        for i in range(self.layers-4):
            model = getattr(self, 'up'+str(i))
            center_out = model.forward(torch.cat([center_out, middle[self.layers-5-i]], 1))

        scale = 1.0
        result= []
        deconv4 = self.deconv4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        output4 = scale * self.output4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        result.append(output4)
        print('output4:',output4.shape)
        deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        output3 = scale * self.output3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        result.append(output3)
        print('output3: ', output3.shape)
        deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        output2 = scale * self.output2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        result.append(output2)
        print('output2: ', output2.shape)
        output1 = scale * self.output1.forward(torch.cat([deconv2, self.upsample(output2)], 1))
        result.append(output1)
        print('output1: ', output1.shape)

        return result
import numpy as np
if __name__ == '__main__':
    model = DepthNet()
    img = torch.rand(1,3,352,640)

    model(img)
