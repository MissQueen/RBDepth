import torch.nn as nn
from models.EdgeExtract import *
from models.DepthNet import *
from models.EdgeExtract import *

class model(nn.Module):
    def __init__(self, Encoder=None, num_features=2048, block_channel=64):
        super(model, self).__init__()
        ori_model = ResNet(Bottleneck, [3, 4, 6, 3])
        self.Encoder = E_resnet(ori_model)
        self.BUBF = BUBF(num_features, block_channel)
        self.DP = DepthNet()

    def forward(self, x):
        b1, b2, b3, b4 = self.Encoder(x)
        bd = self.BUBF(b1, b2, b3, b4)
        out = self.DP(x,bd)
        return out

if __name__=='__main__':
    img = torch.rand(1, 3, 352, 640)
    md = model()
    dp = md(img)