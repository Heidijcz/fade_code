import os
import sys
import numpy
import sys

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image

from data.data_loader import ValDataset

from model.backbone_train import fade
from utils.utils import AverageMeter, pad_img
from option.option_test import opt
from metrics.metric import psnr, ssim

def eval(loader_eval, net, device):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()
    net.eval()
    for batch in tqdm(loader_eval, desc='evaluation'):
        snow_img = batch['snow'].to(device)
        clear_img = batch['clear'].to(device)
        with torch.no_grad():
            H, W = snow_img.shape[2:]
            snow_img = pad_img(snow_img, 16)
            out, out_grad = net(snow_img)
            out = out.clamp(0, 1)
            out = out[:, :, :H, :W]
            if opt.save_infer_results:
                save_image(out, os.path.join(opt.saved_infer_dir, batch['filename'][0]))
        psnr_tmp = psnr(out, clear_img).item()
        ssim_tmp = ssim(out, clear_img).item()
        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)
    return PSNR.avg, SSIM.avg

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = fade(channel=40).to(device)
    test_set = ValDataset(opt.val_snow_dir, opt.val_gt_dir)
    loader_val = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)

    ckpt = torch.load(opt.pre_trained_model, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    net.load_state_dict(ckpt)

    avg_psnr, avg_ssim = eval(loader_val, net, device)
    print('Evaluation on {}\nPSNR:{}\nSSIM:{}'.format(opt.dataset, avg_psnr, avg_ssim))
