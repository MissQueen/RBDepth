
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.DilatedResnet import *

class SSP(nn.Module):#Strip Spatial Perception
    def __init__(self,inchannels,midchannels=21, k=11, w=3):
        super(SSP,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=(k, w), stride=1,
                               padding=(5, 1))
        self.conv2 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=(w, k), stride=1,
                               padding=(1, 5))
        self.conv5 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(num_features=inchannels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        b1 = self.conv1(x)
        b2 = self.conv2(x)
        x = b1 + b2
        x = self.relu(self.bn(self.conv5(x)))

        return x
class RP(nn.Module): #Residual prediction
    def __init__(self, block_channel=184):
        super(RP, self).__init__()
        num_features = 64 + block_channel
        self.conv0 = nn.Conv2d(num_features, num_features,kernel_size=5, stride=1, padding=2, bias=False)

        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,kernel_size=5, stride=1, padding=2, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x0 = self.conv0(x)

        x0 = self.bn0(x0)
        x0 = self.relu(x0)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = x + x1
        x2 = self.conv2(x1)

        return x2
class SRM(nn.Module):#Stripe Refinement
    def __init__(self,num_feature):
        super(SRM,self).__init__()
        self.ssp = SSP(64+num_feature//32)
        self.R = RP(num_feature//32)

    def forward(self,x_decoder,x_bubf):
        out = self.R(self.ssp(torch.cat((x_decoder, x_bubf), 1)))
        return out


class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out

class E_resnet(nn.Module):
    def __init__(self, original_model, num_features=2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)
        return x_block1, x_block2, x_block3, x_block4

class lRB(nn.Module):   #large Eefinement Block
    def __init__(self, in_channels, out_channels):
        super(lRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.conv1(x)

class BUBF(nn.Module):  #Bottom-Up Boundary Fusion
    def __init__(self, channels, out_channel):
        super(BUBF, self).__init__()
        self.lrb_1 = lRB(channels//8, out_channel)
        self.lrb_2 = lRB(channels//4, out_channel)
        self.lrb_3 = lRB(channels//2, out_channel)
        self.lrb_4 = lRB(channels, out_channel)
        self.lrb_5 = lRB(out_channel, out_channel)
        self.lrb_6 = lRB(out_channel, out_channel)
        self.lrb_7 = lRB(out_channel, out_channel)

        self.up1 = _UpProjection(out_channel, out_channel)
        self.up2 = _UpProjection(out_channel, out_channel)
        self.up3 = _UpProjection(out_channel, out_channel)
        self.up4 = _UpProjection(out_channel, out_channel)
    def forward(self, x_block1, x_block2, x_block3, x_block4):
        print(x_block1.shape, x_block2.shape, x_block3.shape, x_block4.shape)
        x1 = self.lrb_1(x_block1)
        # print('BUBF x1:', x1.shape)
        x1 = self.up4(x1, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        print('BUBF x1:',x1.shape)
        x2 = self.lrb_2(x_block2)
        # print('BUBF x2:', x2.shape)
        x2 = self.up1(x2, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        print('BUBF up1:', x2.shape)
        x2 = x1 + x2
        x2 = self.lrb_5(x2)
        print('BUBF x2:', x2.shape)
        x3 = self.lrb_3(x_block3)
        # print('BUBF x3:', x3.shape)
        x3 = self.up2(x3, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        # print('BUBF up2:', x3.shape)
        x3 = x2 + x3
        x3 = self.lrb_6(x3)
        print('BUBF x3:', x3.shape)
        x4 = self.lrb_4(x_block4)
        # print('BUBF x4:', x4.shape)
        x4 = self.up3(x4, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        # print('BUBF up3:', x4.shape)
        x4 = x3 + x4
        x4 = self.lrb_7(x4)
        print('BUBF x4:', x4.shape)
        return x4

import numpy as np

if __name__ == '__main__':
    number_features = 2048
    img = np.random.randint(0,255,size=(1, 3, 640 , 352))
    # print(img)
    img = np.asarray(img, np.float32)
    img = torch.from_numpy(img)
    ori_model = ResNet(Bottleneck, [3, 4, 6, 3])
    Encoder = E_resnet(ori_model)
    b1, b2, b3, b4 = Encoder(img)
    bubf = BUBF(number_features, 64)
    out = bubf(b1,b2,b3,b4)


