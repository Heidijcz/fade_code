from torch import nn
import numpy as np
import torch
import scipy.linalg
import torch.nn.functional as F
import torch.nn.init as init

from .deconv import DEConv


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def thopspixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


def thopsum(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


class MCBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True):
        super(MCBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=1, dilation=1, bias=bias)
        self.conv3 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=2, dilation=2, bias=bias)
        self.conv4 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=4, dilation=4, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(n_feats, n_feats, 1, bias=bias)

        self.fusion = nn.Conv2d(n_feats * 5, n_feats, 1, bias=bias)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = self.act(self.conv3(x))
        x4 = self.act(self.conv4(x))

        x_global = self.avg_pool(x)
        x_global = self.act(self.conv_global(x_global))
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=False)

        out = self.fusion(torch.cat([x1, x2, x3, x4, x_global], dim=1))
        return out + x


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thopspixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float() \
                    .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thopsum(self.log_s) * thopspixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d=1, clamp=0.8):
        super(InvBlock, self).__init__()
        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        out = torch.cat((y1, y2), 1)

        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale 
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d=1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1,
            use_separable_second_conv=False):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            if i == 1 and use_separable_second_conv:
                modules_body.append(nn.Sequential(
                    nn.Conv2d(
                        n_feat, n_feat, kernel_size,
                        padding=kernel_size // 2, groups=n_feat, bias=bias),
                    nn.Conv2d(n_feat, n_feat, 1, bias=bias)
                ))
            else:
                modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction, n_resblocks,
            use_separable_second_conv=False):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1,
                use_separable_second_conv=use_separable_second_conv) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class get_Fre(nn.Module):
    def __init__(self):
        super(get_Fre, self).__init__()

    def forward(self, dp):
        dp = torch.fft.rfft2(dp, norm='backward')
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)

        return dp_amp, dp_pha


