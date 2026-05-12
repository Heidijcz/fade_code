from torch import nn
import torch

from .modules.gradnet import GRM
from .modules.frequencynet import FRM
from .modules.modules import default_conv, ResBlock, ResidualGroup

class fade(nn.Module):
    def __init__(self, channel):
        super(fade, self).__init__()
        self.gradNet = GRM(channel)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channel,kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.rb1 = ResBlock(default_conv, channel, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rb2 = ResBlock(default_conv, channel, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rb3 = ResBlock(default_conv, channel, 3, bias=True, bn=False,
                                 act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        rbm = [
            ResidualGroup(default_conv, 3 * channel, 3, reduction=16, n_resblocks=4,
                          use_separable_second_conv=True),
            ResidualGroup(default_conv, 3 * channel, 3, reduction=16, n_resblocks=4,
                          use_separable_second_conv=True),
            ResidualGroup(default_conv, 3 * channel, 3, reduction=16, n_resblocks=3,
                          use_separable_second_conv=True)
        ]
        self.rbm = nn.Sequential(*rbm)

        self.conv2 = default_conv(7 * channel, 3 * channel, 1)
        last_conv = [
            default_conv(3 * channel, channel, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(channel, 3, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)

        self.frequencyNet1 = FRM(channel,channel)
        self.frequencyNet2 = FRM(channel*2, channel)
        self.frequencyNet3 = FRM(channel*4, channel)

    def forward(self, x):
        Grad_out, F_refined = self.gradNet(x) 

        F_i1in = self.rb1(self.act(self.conv1(x)))

        F_grad1out, F_i1out = self.frequencyNet1(F_i1in, F_refined)

        F_i2in = self.rb2(F_i1out)
        F_grad2in = torch.cat((F_grad1out, F_refined), 1)
        F_grad2out, F_i2out = self.frequencyNet2(F_i2in, F_grad2in)

        F_i3in = self.rb3(F_i2out)
        F_grad3in = torch.cat((F_grad2out, F_grad2in), 1)
        F_grad3out, F_i3out = self.frequencyNet3(F_i3in, F_grad3in)

        F_combine = torch.cat([F_i1out, F_i2out, F_i3out, F_grad3out], 1)
        F_combine_m = self.rbm(self.conv2(F_combine))
        out = self.last_conv(F_combine_m) + x
        return out, Grad_out





























