from torch import nn
import torch
from .modules import InvBlock, DenseBlock, MCBlock

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class PGBlock(nn.Module):
    def __init__(self, channels, rgb_channels):
        super(PGBlock, self).__init__()
        self.pre_rgb = nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        self.pre_grad = nn.Conv2d(channels, rgb_channels, 1, 1, 0)

        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0),
            nn.Sigmoid()
        )

        self.rb = nn.Sequential(
            nn.Conv2d(2 * rgb_channels, rgb_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        )

    def forward(self, grad_pha, rgb_pha):
        grad1 = self.pre_grad(grad_pha)
        rgb1 = self.pre_rgb(rgb_pha)

        gate_map = self.gate(grad1)
        rgb_filtered = rgb1 * gate_map

        pha_combine = torch.cat([grad1, rgb_filtered], 1)
        out = self.rb(pha_combine)

        return out + grad1

class ARBlock(nn.Module):
    def __init__(self, channels, rgb_channels):
        super(ARBlock, self).__init__()
        self.pre_rgb = nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        self.pre_grad = nn.Conv2d(channels, rgb_channels, 1, 1, 0)

        self.rb = nn.Sequential(
            nn.Conv2d(2 * rgb_channels, rgb_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        )

    def forward(self, grad_amp, rgb_amp):
        grad1 = self.pre_grad(grad_amp)
        rgb1 = self.pre_rgb(rgb_amp)

        amp_combine = torch.cat([grad1, rgb1], 1)
        out = self.rb(amp_combine)

        return rgb1 + out

class SDB(nn.Module):
    def __init__(self, channel, rgb_channel):
        super(SDB, self).__init__()
        self.pre1 = nn.Conv2d(channel, channel, 1, 1, 0)
        self.pre2 = nn.Conv2d(rgb_channel, rgb_channel, 1, 1, 0)
        self.arb = ARBlock(channel, rgb_channel)
        self.pgb = PGBlock(channel, rgb_channel)
        self.conv = nn.Conv2d(rgb_channel, rgb_channel, 1, 1, 0)

    def forward(self, rgb, grad):
        _, _, H, W = grad.shape
        grad = torch.fft.rfft2(self.pre1(grad)+1e-8, norm='backward')
        rgb = torch.fft.rfft2(self.pre2(rgb) + 1e-8, norm='backward')
        grad_amp = torch.abs(grad)
        grad_pha = torch.angle(grad)
        rgb_amp = torch.abs(rgb)
        rgb_pha = torch.angle(rgb)
        amp_res = self.arb(grad_amp,rgb_amp)
        pha_res = self.pgb(grad_pha, rgb_pha)
        real = amp_res * torch.cos(pha_res) + 1e-8
        imag = amp_res * torch.sin(pha_res) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        return self.conv(out)

class FRM(nn.Module):
    def __init__(self, channel, rgb_channel):
        super(FRM, self).__init__()
        self.grad_conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.grad_conv2 = nn.Conv2d(channel, channel, 1, 1, 0)
        self.rgb_conv1 = nn.Conv2d(rgb_channel, rgb_channel, 3, 1, 1)
        self.spa = nn.Sequential(InvBlock(DenseBlock, channel + rgb_channel, rgb_channel),
                                         MCBlock(channel + rgb_channel),
                                         nn.Conv2d(channel + rgb_channel, rgb_channel, 1, 1, 0))
        self.fre = SDB(channel, rgb_channel)
        self.fuse = nn.Sequential(InvBlock(DenseBlock, 2 * rgb_channel, rgb_channel),
                                          nn.Conv2d(2 * rgb_channel, rgb_channel, 1, 1, 0))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.contrast = stdv_channels
        self.hpa = nn.Sequential(nn.Conv2d(rgb_channel, rgb_channel // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(rgb_channel // 2, rgb_channel, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())

    def forward(self, rgb_in, grad_in):
        grad_out = self.act(self.grad_conv1(grad_in))
        grad_mapping = self.grad_conv2(grad_out)
        rgb_mapping = self.rgb_conv1(rgb_in)
        F_spa = self.spa(torch.cat([rgb_mapping, grad_mapping], dim=1))
        F_freq = self.fre(rgb_mapping, grad_mapping)
        F_cat = self.fuse(torch.cat([F_spa, F_freq], 1))
        att_map = self.contrast(F_cat) + self.avgpool(F_cat) + self.maxpool(F_cat)
        rgb_out = self.hpa(att_map) * F_cat
        return grad_out, rgb_out

