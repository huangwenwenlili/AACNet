import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.external_function import SpectralNorm
from .base_function import init_net
import numbers
import math


"""
def define_g(init_type='normal', gpu_ids=[]):
    net = InpaintGenerator()
    return init_net(net, init_type, gpu_ids)
"""

def define_e(init_type='normal', gpu_ids=[]):
    net = Encoder()
    return init_net(net, init_type, gpu_ids)

def define_g(init_type='normal', gpu_ids=[]):
    net = InpaintGenerator()
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)

def define_d_msg(init_type= 'normal', gpu_ids=[]):
    net = Discriminator_msg(in_channels=3)
    return init_net(net, init_type, gpu_ids)

class InpaintGenerator(nn.Module):
    def __init__(self, ngf=64):
        super(InpaintGenerator, self).__init__()



    def forward(self, img_m, mask):
        x=img_m
        return x




class Encoder(nn.Module):
    def __init__(self, ngf=64):
        super(Encoder, self).__init__()

        self.down0 = RefineBlcok(in_ch=3, out_ch=ngf, kernel_size=5, stride=1, padding=2, img_size=256)
        self.down1 = RefineBlcok(in_ch=ngf, out_ch=ngf*2, kernel_size=3, stride=2, padding=1, img_size=128)
        self.down11 = RefineBlcok(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1, img_size=128)
        self.down2 = RefineBlcok(in_ch=ngf*2, out_ch=ngf*4, kernel_size=3, stride=2, padding=1, img_size=128)



    def forward(self, img_m, mask):
        x, m = self.down0(img_m, mask)
        x, m = self.down1(x, m)

        x, m = self.down11(x, m)
        x, m = self.down2(x, m)


        return x, m


class Decoder(nn.Module):
    def __init__(self, ngf=64):
        super(Decoder, self).__init__()
        self.middle1 = RefineBlcok(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1)
        self.middle2 = RefineBlcok(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=2, padding=2)
        self.middle3 = RefineBlcok(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=3, padding=3)
        self.middle4 = RefineBlcok(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1)
        self.middle5 = RefineBlcok(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=2, padding=2)
        self.middle6 = RefineBlcok(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=3, padding=3)
        self.middle7 = RefineBlcok(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1)

        self.up1 = RefineBlcok(in_ch=ngf*4, out_ch=ngf*2, kernel_size=3, stride=1, padding=1, img_size=128)
        self.up11 = RefineBlcok(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1, img_size=128)
        self.up2 = RefineBlcok(in_ch=ngf*2, out_ch=ngf, kernel_size=3, stride=1, padding=1, img_size=128)
        self.up21 = RefineBlcok(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1, img_size=128)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, feature, mask):
        x, m = self.middle1(feature, mask)
        x, m = self.middle2(x, m)
        x, m = self.middle3(x, m)
        x, m = self.middle4(x, m)
        x, m = self.middle5(x, m)
        x, m = self.middle6(x, m)
        x, m = self.middle7(x, m)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        x, m = self.up1(x, m)
        x, m = self.up11(x, m)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        x, m = self.up2(x, m)
        x, m = self.up21(x, m)
        x = self.out(x)

        return x

class Discriminator_msg(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(Discriminator_msg, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64+3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128+3, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256+3, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=4, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )#dy outchannels=1 patchgan

    def forward(self, x):
        conv1 = self.conv1(x['x_out_L256'])
        conv1 = torch.cat([conv1, x['x_out_L128']], dim=1)
        conv2 = self.conv2(conv1)
        conv2 = torch.cat([conv2, x['x_out_L64']], dim=1)
        conv3 = self.conv3(conv2)
        conv3 = torch.cat([conv3, x['x_out_L32']], dim=1)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Discriminator(nn.Module):
    def __init__(self, in_channels, use_spectral_norm=True):
        super(Discriminator, self).__init__()

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

        x = torch.sigmoid(conv5)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

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
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.InstanceNorm2d(out_ch, track_running_stats=False)

    def forward(self, x):
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + residual

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class RefineConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, img_size=256):
        super().__init__()
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)

        self.act = nn.PReLU()
        if in_channels != 3:
            self.mask_conv2d = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                nn.InstanceNorm2d(out_channels, track_running_stats=False),
                nn.Sigmoid()
            )
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.gus = GaussianSmoothing(out_channels, kernel_size=3, img_size=img_size)
        else:
            self.mask_conv2d = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(1, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias),
                nn.InstanceNorm2d(out_channels, track_running_stats=False),
                nn.Sigmoid()
            )
            self.conv2d = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias)
            )
            self.gus = GaussianSmoothing(out_channels, kernel_size=7, img_size=img_size)

    def forward(self, feature, mask):
        x = self.conv2d(feature)
        m = self.mask_conv2d(mask)
        x = x * m + (1 - m) * self.gus(x)
        x = self.act(self.norm(x))
        return x, m


class RefineBlcok(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1, img_size=64):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act1 = nn.PReLU()
        self.gus = GaussianSmoothing(out_ch, kernel_size=kernel_size, img_size=img_size)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        if dilation == 1:
            self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        else:
            self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        if in_ch != 3:
            self.mask_conv2d = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation),
                nn.InstanceNorm2d(out_ch, track_running_stats=False),
                nn.Sigmoid()
            )
        else:
            self.mask_conv2d = nn.Sequential(
                nn.Conv2d(1, out_ch, kernel_size, stride, padding, dilation),
                nn.InstanceNorm2d(out_ch, track_running_stats=False),
                nn.Sigmoid()
            )

    def forward(self, x, mask):
        m = self.mask_conv2d(mask)
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.refine(out, m)
        out = out + residual
        return out, m

    def refine(self, x, mask):

        out = x * mask
        temp = self.gus(out)
        coe = self.avg(mask) + 1e-4
        temp = temp / coe
        out = out + (1 - mask) * temp

        return out


class RefineFuse(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1, img_size=64):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act1 = nn.PReLU()
        self.gus = GaussianSmoothing(out_ch, kernel_size=kernel_size, img_size=img_size)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        if dilation == 1:
            self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        else:
            self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.mask_conv2d = nn.Sequential(
            nn.Conv2d(out_ch+1, out_ch, kernel_size, stride, padding, dilation),
            nn.InstanceNorm2d(out_ch, track_running_stats=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask, mask2):
        m = self.mask_conv2d(torch.cat([mask, mask2], dim=1))
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.refine(out, m)
        out = out + residual
        return out, m

    def refine(self, x, mask):

        out = x * mask
        temp = self.gus(out)
        coe = self.avg(mask) + 1e-4
        temp = temp / coe
        out = out + (1 - mask) * temp

        return out







class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
    """
    def __init__(self, channels, kernel_size=3, sigma=1, dim=2, img_size=256):
        super(GaussianSmoothing, self).__init__()

        self.pad = nn.ReflectionPad2d(get_pad(img_size, ksize=kernel_size, stride=1))
        if isinstance(kernel_size, numbers.Number):
            self.kernel_size = kernel_size
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))


        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        x = self.pad(x)
        return self.conv(x, weight=self.weight, groups=self.groups)


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


