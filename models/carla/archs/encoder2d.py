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

class Encoder2d(nn.Module):
    def __init__(self, in_dim=32, mid_dim=64, out_dim=32):
        super().__init__()

        self.encoder_layer0 = Res2dBlock(in_dim, mid_dim)
        self.encoder_layer1 = Pool2dBlock(2)
        self.encoder_layer2 = Res2dBlock(mid_dim, mid_dim)
        self.encoder_layer3 = Res2dBlock(mid_dim, mid_dim)
        self.encoder_layer4 = Res2dBlock(mid_dim, mid_dim)
        self.encoder_layer5 = Pool2dBlock(2)
        self.encoder_layer6 = Res2dBlock(mid_dim, mid_dim)
        self.encoder_layer7 = Res2dBlock(mid_dim, mid_dim)
        self.encoder_layer8 = Res2dBlock(mid_dim, mid_dim)
        self.final_layer = nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.encoder_layer0(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.encoder_layer5(x)
        x = self.encoder_layer6(x)
        x = self.encoder_layer7(x)
        x = self.encoder_layer8(x)
        x = self.final_layer(x)
        return x

class Net2d(nn.Module):
    def __init__(self, in_chans, mid_chans=64, out_chans=1):
        super(Net2d, self).__init__()
        conv2d = []
        conv2d_transpose = []
        up_bn = []

        self.down_in_dims = [in_chans, mid_chans, 2*mid_chans]
        self.down_out_dims = [mid_chans, 2*mid_chans, 4*mid_chans]
        self.down_ksizes = [3, 3, 3]
        self.down_strides = [2, 2, 2]
        padding = 1

        for i, (in_dim, out_dim, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            conv2d.append(nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm2d(num_features=out_dim),
            ))
        self.conv2d = nn.ModuleList(conv2d)

        self.up_in_dims = [4*mid_chans, 6*mid_chans]
        self.up_bn_dims = [6*mid_chans, 3*mid_chans]
        self.up_out_dims = [4*mid_chans, 2*mid_chans]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        padding = 1 # Note: this only holds for ksize=4 and stride=2!
        print('up dims: ', self.up_out_dims)

        for i, (in_dim, bn_dim, out_dim, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
            conv2d_transpose.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
            ))
            up_bn.append(nn.BatchNorm2d(num_features=bn_dim))

        # final 1x1x1 conv to get our desired out_chans
        self.final_feature = nn.Conv2d(in_channels=3*mid_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)
        self.conv2d_transpose = nn.ModuleList(conv2d_transpose)
        self.up_bn = nn.ModuleList(up_bn)
        
    def forward(self, inputs):
        feat = inputs
        skipcons = []
        for conv2d_layer in self.conv2d:
            feat = conv2d_layer(feat)
            skipcons.append(feat)

        skipcons.pop() # we don't want the innermost layer as skipcon

        for i, (conv2d_transpose_layer, bn_layer) in enumerate(zip(self.conv2d_transpose, self.up_bn)):
            feat = conv2d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) # skip connection by concatenation
            feat = bn_layer(feat)

        feat = self.final_feature(feat)

        return feat

if __name__ == "__main__":
    net = Net2d(in_chans=4, mid_chans=32, out_chans=3)
    print(net.named_parameters)
    inputs = torch.rand(2, 4, 128, 384)
    out = net(inputs)
    print(out.size())


