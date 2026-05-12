import torch.nn as nn
from torchaudio.functional import contrast
from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
    def forward(self, out, label, input):
        out_vgg, label_vgg, input_vgg = self.vgg(out), self.vgg(label), self.vgg(input)
        loss = 0

        d_outlabel, d_outinput = 0, 0
        for i in range(len(out_vgg)):
            d_outlabel = self.l1(out_vgg[i], label_vgg[i].detach())
            if not self.ab:
                d_outinput = self.l1(out_vgg[i], input_vgg[i].detach())
                contrastive = d_outlabel / (d_outinput + 1e-7)
            else:
                contrastive = d_outlabel
            loss += contrastive * self.weights[i]
        return loss



