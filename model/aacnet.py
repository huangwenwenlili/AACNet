from torch import nn
import torch
import math
from torch.nn import functional as F
from model.base_function import init_net
import numbers
import numpy as np
from model.dat_blocks import DAttentionBaseline_gate_factor
from timm.models.layers import to_2tuple, trunc_normal_

def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(ngf=48)
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)

class Generator(nn.Module):
    def __init__(self, ngf=48):
        super().__init__()

        self.start = ResBlock0_v2(in_ch=4, out_ch=ngf, kernel_size=5, stride=1, padding=2)

        self.trane256 = ResBlock_v2(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)

        self.down128 = Downsample(num_ch=ngf) # B *2ngf * 128, 128
        self.trane128 = nn.Sequential(
            ResBlock_v2(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf * 2, out_ch=ngf * 2, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf * 2, out_ch=ngf * 2, kernel_size=3, stride=1, padding=1),
        )
        self.down64 = Downsample(num_ch=ngf*2) # B *4ngf * 64, 64
        self.trane64 = nn.Sequential(
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
        )
        self.down32 = Downsample(num_ch=ngf*4)  # B *8ngf * 32, 32
        self.middle1 = ResBlock_v2(in_ch=ngf * 8, out_ch=ngf * 8, kernel_size=3, stride=1, padding=1)
        fmap_size = to_2tuple(32)
        heads=12
        hc = int(ngf * 8/heads)
        n_groups = int(3)
        attn_drop=0.0
        proj_drop=0.0
        stride = 1
        offset_range_factor=2
        use_pe=True
        dwc_pe=False
        no_off = False
        fixed_pe = False
        stage_idx=2

        self.attn1 = DAttentionBaseline_gate_factor(fmap_size, fmap_size, heads,
                    hc, n_groups, attn_drop, proj_drop,
                    stride, offset_range_factor, use_pe, dwc_pe,
                    no_off, fixed_pe, stage_idx)

        self.middle2 = ResBlock_v2(in_ch=ngf * 8, out_ch=ngf * 8, kernel_size=3, stride=1, padding=1)
        self.attn2 = DAttentionBaseline_gate_factor(fmap_size, fmap_size, heads,
                    hc, n_groups, attn_drop, proj_drop,
                    stride, offset_range_factor, use_pe, dwc_pe,
                    no_off, fixed_pe, stage_idx)
        self.middle3 = ResBlock_v2(in_ch=ngf * 8, out_ch=ngf * 8, kernel_size=3, stride=1, padding=1)
        self.attn3 = DAttentionBaseline_gate_factor(fmap_size, fmap_size, heads,
                    hc, n_groups, attn_drop, proj_drop,
                    stride, offset_range_factor, use_pe, dwc_pe,
                    no_off, fixed_pe, stage_idx)
        self.middle4 = ResBlock_v2(in_ch=ngf * 8, out_ch=ngf * 8, kernel_size=3, stride=1, padding=1)
        self.attn4 = DAttentionBaseline_gate_factor(fmap_size, fmap_size, heads,
                    hc, n_groups, attn_drop, proj_drop,
                    stride, offset_range_factor, use_pe, dwc_pe,
                    no_off, fixed_pe, stage_idx)
        self.middle5 = ResBlock_v2(in_ch=ngf * 8, out_ch=ngf * 8, kernel_size=3, stride=1, padding=1)
        in_c = 3
        self.x_out_L32 = nn.Conv2d(ngf * 8, in_c, 3, 1, 1)  # 32

        self.up64 = Upsample(ngf*8)  # B *4ngf * 64, 64
        self.fuse64 = nn.Conv2d(in_channels=ngf*4*2, out_channels=ngf*4, kernel_size=1, stride=1, bias=False)
        self.trand64 = nn.Sequential(
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1),
        )
        self.x_out_L64 = nn.Conv2d(ngf*4, in_c, 3, 1, 1)  # 64

        self.up128 = Upsample(ngf*4) # B *2ngf * 128, 128
        self.fuse128 = nn.Conv2d(in_channels=4*ngf, out_channels=2*ngf, kernel_size=1, stride=1, bias=False)
        self.trand128 = nn.Sequential(
            ResBlock_v2(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1),
            ResBlock_v2(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1),
        )
        self.x_out_L128 = nn.Conv2d(ngf*2, in_c, 3, 1, 1)  # 128

        self.up256 = Upsample(ngf*2) # B *ngf * 256, 256
        self.fuse256 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=1, stride=1)
        self.trand256 = ResBlock_v2(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)

        self.trand2562 = ResBlock_v2(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0)
        )


    def forward(self, x, mask=None):
        noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 128.))
        x = x + noise
        feature = torch.cat([x, mask], dim=1)
        x_outs = {}
        feature256 = self.start(feature)
        #m = F.interpolate(mask, size=feature.size()[-2:], mode='nearest')
        feature256 = self.trane256(feature256)
        feature128 = self.down128(feature256)
        feature128 = self.trane128(feature128)
        feature64 = self.down64(feature128)
        feature64 = self.trane64(feature64)
        feature32 = self.down32(feature64)

        feature32 = self.middle1(feature32)
        bs, c, h, w = feature32.shape

        feature321,pos,ref = self.attn1(feature32)
        feature32 = feature32 + feature321

        feature32 = self.middle2(feature32)
        feature321,pos,ref = self.attn2(feature32)
        feature32 = feature32 + feature321

        feature32 = self.middle3(feature32)
        feature321, pos,ref = self.attn3(feature32)
        feature32 = feature32 + feature321

        feature32 = self.middle4(feature32)
        feature321, pos,ref = self.attn4(feature32)
        feature32 = feature32 + feature321

        feature32 = self.middle5(feature32)
        x_outs['x_out_L32'] = torch.tanh(self.x_out_L32(feature32))

        out64 = self.up64(feature32)
        out64 = self.fuse64(torch.cat([feature64, out64], dim=1))
        out64 = self.trand64(out64)
        x_outs['x_out_L64'] = torch.tanh(self.x_out_L64(out64))
        #out128 = torch.nn.functional.interpolate(out64, scale_factor=2, mode='nearest')
        out128 = self.up128(out64)
        out128 = self.fuse128(torch.cat([feature128, out128], dim=1))
        out128 = self.trand128(out128)
        x_outs['x_out_L128'] = torch.tanh(self.x_out_L128(out128))

        out256 = self.up256(out128)
        out256 = self.fuse256(torch.cat([feature256, out256], dim=1))
        out256 = self.trand256(out256)
        #out256 = self.trand2562(out256)
        out = torch.tanh(self.out(out256))
        x_outs['x_out_L256'] = out
        return out, x_outs

class ResBlock0_v2(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1,gate=False):
        super().__init__()
        gate = True
        self.gate = gate
        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act1 = nn.LeakyReLU(0.2, True)
        if self.gate:
            self.conv2_ga = GatedConv(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1,
                                      padding=padding, dilation=dilation)
        else:
            self.conv2_org = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding,
                                       dilation=dilation)

    def forward(self, x):
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act1(out)
        if self.gate:
            out = self.conv2_ga(out)
        else:
            out = self.conv2_org(out)

        out = out + residual

        return out

class ResBlock_v2(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1,gate=False):
        super().__init__()
        gate = True
        self.gate = gate
        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        if self.gate:
            self.conv2_ga = GatedConv(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        else:
            self.conv2_org = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding,
                                  dilation=dilation)
        self.n0 = nn.InstanceNorm2d(in_ch, track_running_stats=False)

    def forward(self, x):
        residual = self.projection(x)
        out = self.n0(x)
        out = self.act0(out)
        out = self.conv1(out)
        out = self.n1(out)
        out = self.act1(out)
        if self.gate:
            out = self.conv2_ga(out)
        else:
            out = self.conv2_org(out)
        out = out + residual

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]




class Downsample(nn.Module):
    def __init__(self, num_ch=32):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch*2, track_running_stats=False),
            nn.GELU()
        )

        #self.body = nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)



class Upsample(nn.Module):
    def __init__(self, num_ch=32):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch//2, track_running_stats=False),
            nn.GELU()
        )

        #self.body = nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.body(x)

#
# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1):
#         super().__init__()
#
#         if out_ch is None or out_ch == in_ch:
#             out_ch = in_ch
#             self.projection = nn.Identity()
#         else:
#             self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)
#
#         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
#         self.act1 = nn.GELU()
#         self.act2 = nn.GELU()
#         self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
#         self.n2 = nn.InstanceNorm2d(in_ch, track_running_stats=False)
#
#     def forward(self, x):
#         residual = self.projection(x)
#         out = self.conv1(x)
#         out = self.n1(out)
#         out = self.act1(out)
#         out = self.conv2(out)
#         out = self.n2(out)
#         out = out + residual
#         out = self.act2(out)
#         return out

class ResBlock0(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act1 = nn.LeakyReLU(0.2, True)
        self.conv2 = GatedConv(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)


    def forward(self, x):
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + residual

        return out

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = GatedConv(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n0 = nn.InstanceNorm2d(in_ch, track_running_stats=False)

    def forward(self, x):
        residual = self.projection(x)
        out = self.n0(x)
        out = self.act0(out)
        out = self.conv1(out)
        out = self.n1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + residual

        return out

class GatedConv(nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * torch.sigmoid(mask)
        return x

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module