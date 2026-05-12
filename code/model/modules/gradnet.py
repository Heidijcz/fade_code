from torch import nn
import torch.nn.functional as F
import torch
import torchvision.ops

from .modules import default_conv, ResBlock, MCBlock
from .deconv import DEConv



class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class GEBlock(nn.Module):
    def __init__(self, conv, channel, kernel_size):
        super(GEBlock, self).__init__()
        self.conv1 = DEConv(channel)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(channel, channel, kernel_size, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = out + x
        out = self.conv2(out)
        out = out + x
        return out

class SRBlock(nn.Module):
    def __init__(self, channel):
        super(SRBlock, self).__init__()
        self.offset_conv = nn.Conv2d(channel, 2 * 3 * 3, kernel_size=3, padding=1)
        self.dcn = torchvision.ops.DeformConv2d(channel, channel, kernel_size=3, padding=1)

        self.gate_net = nn.Sequential(
            default_conv(channel, channel // 2, 1),
            nn.ReLU(True),
            default_conv(channel // 2, channel, 1),
            nn.Sigmoid()
        )

        self.conv_out = default_conv(channel, channel, 3)

    def forward(self, x):
        offsets = self.offset_conv(x)
        F_align = self.dcn(x, offsets)

        gate = self.gate_net(F_align)
        out = F_align * gate
        
        out = self.conv_out(out)
        return out


class GRM(nn.Module):
    def __init__(self, channel):
        super(GRM, self).__init__()
        self.grad = Get_gradient()
        self.grm_conv1 = default_conv(3, channel, 3)

        self.geb = GEBlock(default_conv, channel, 3)
        self.mcb = MCBlock(channel)

        self.srb = SRBlock(channel)

        self.conv_out = default_conv(channel, 3, 3)

    def forward(self, x):
        grad_snowy = self.grad(x)
        F_grad = self.grm_conv1(grad_snowy)

        F_geb = self.geb(F_grad)

        F_mcb = self.mcb(F_geb) + F_geb

        F_refined = self.srb(F_mcb)

        Grad_out = self.conv_out(F_refined)
        
        return Grad_out, F_refined
