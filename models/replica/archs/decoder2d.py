import torch
import torch.nn as nn
import torch.nn.functional as F

class Res2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, ksize=3, padding=1):
        super(Res2dBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=1, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=padding),
            nn.BatchNorm2d(out_planes)
        )
        assert(padding==1 or padding==0)
        self.padding = padding
        self.ksize = ksize

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        # print('res', res.shape)
        skip = self.skip_con(x)
        if self.padding==0 and self.ksize==3:
            # the data has shrunk a bit
            skip = skip[:,:,2:-2,2:-2]
        # print('skip', skip.shape)
        # # print('trim', skip.shape)
        return F.relu(res + skip, True)

class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)
    
class Pool2dBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool2dBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)

class Decoder2d(nn.Module):
    def __init__(self, in_dim=32, mid_dim=64, out_dim=32):
        super().__init__()

        self.decoder_layer0 = Res2dBlock(in_dim, mid_dim)
        self.decoder_layer1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_layer2 = Res2dBlock(mid_dim, mid_dim)
        self.decoder_layer3 = Res2dBlock(mid_dim, mid_dim)
        self.decoder_layer4 = Res2dBlock(mid_dim, mid_dim)
        self.decoder_layer5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_layer6 = Res2dBlock(mid_dim, mid_dim)
        self.decoder_layer7 = Res2dBlock(mid_dim, mid_dim)
        self.decoder_layer8 = Res2dBlock(mid_dim, mid_dim)
        self.final_layer = nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.decoder_layer0(x)
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)
        x = self.decoder_layer5(x)
        x = self.decoder_layer6(x)
        x = self.decoder_layer7(x)
        x = self.decoder_layer8(x)
        x = self.final_layer(x)
        return x
