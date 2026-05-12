import os, time, math
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader


from data.data_loader import TrainDataset, TestDataset
from option.option_train import opt
from model.backbone_train import fade
from model.modules.modules import get_Fre
from model.modules.gradnet import Get_gradient

from loss.loss import ContrastLoss
from logger.logger import plot_loss_log, plot_psnr_log, plot_ssim_log
from metrics.metric import psnr, ssim
from utils.utils import pad_img


start_time = time.time()


def set_seed_torch(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5* (init_lr - end_lr) * (1 + math.cos(math.pi * t / T))
    return lr

def train(net, loader_train, loader_test, optimizer, criterion):

    losses = [] 
    LOSS_L1 = []  
    LOSS_CR = []  
    LOSS_GRAD = [] 
    LOSS_AMP = []
    LOSS_PHA = []

    loss_log = {'L1':[], 'CR':[], 'grad':[],'fre_amp':[],'fre_pha':[],'total':[]} 
    loss_log_tmp = {'L1':[], 'CR':[], 'grad':[],'fre_amp':[],'fre_pha':[],'total':[]} 

    psnrs = []
    ssims =[]
    psnr_log =[]
    ssim_log =[]
    eval_epochs = []
    max_ssim = 0
    max_psnr = 0

    get_fre = get_Fre().to(opt.device)
    get_grad = Get_gradient().to(opt.device)

    loader = iter(loader_train)

    start_step = 0
    steps = len(loader_train) * opt.epochs
    for step in range(start_step+1,steps+1):
        net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        input, label = next(loader)
        input = input.to(opt.device)
        label = label.to(opt.device)

        out, out_grad = net(input)
        out_amp, out_pha = get_fre(out)
        gt_amp, gt_pha = get_fre(label)
        gt_grad = get_grad(label)

        loss_L1 = criterion[0](out, label)
        loss_CR = criterion[1](out, label, input)
        loss_grad = criterion[2](out_grad, gt_grad)
        loss_fre_amp = criterion[3](out_amp, gt_amp)
        loss_fre_pha = criterion[4](out_pha, gt_pha)
        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR + opt.w_loss_Grad * loss_grad + opt.w_loss_Pre_amp * loss_fre_amp + opt.w_loss_Pre_pha * loss_fre_pha
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()


        losses.append(loss.item())
        LOSS_L1.append(loss_L1.item())
        LOSS_CR.append(loss_CR.item())
        LOSS_GRAD.append(loss_grad.item())
        LOSS_AMP.append(loss_fre_amp.item())
        LOSS_PHA.append(loss_fre_pha.item())

        loss_log_tmp['L1'].append(loss_L1.item())
        loss_log_tmp['CR'].append(loss_CR.item())
        loss_log_tmp['grad'].append(loss_grad.item())
        loss_log_tmp['fre_amp'].append(loss_fre_amp.item())
        loss_log_tmp['fre_pha'].append(loss_fre_pha.item())
        loss_log_tmp['total'].append(loss.item())
        print(
            f'\rloss:{loss.item():.5f} | L1:{loss_L1.item():.5f} | CR:{loss_CR.item():.5f} | grad:{loss_grad.item():.5f} | pre_amp:{loss_fre_amp.item():.5f} | pre_pha:{loss_fre_pha.item():.5f} | step :{step}/{steps} | lr :{lr :.7f} | time_used :{(time.time()-start_time) / 60 :.1f}', end='', flush=True
        )

        if step % len(loader_train) == 0:
            loader = iter(loader_train)
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            plot_loss_log(loss_log, int(step / len(loader_train)), opt.saved_plot_dir)
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
            np.save(os.path.join(opt.saved_data_dir, 'loss_L1.npy'), LOSS_L1)
            np.save(os.path.join(opt.saved_data_dir, 'loss_CR.npy'), LOSS_CR) 
            np.save(os.path.join(opt.saved_data_dir, 'loss_grad.npy'), LOSS_GRAD)
            np.save(os.path.join(opt.saved_data_dir, 'loss_amp.npy'), LOSS_AMP)
            np.save(os.path.join(opt.saved_data_dir, 'loss_pha.npy'), LOSS_PHA) 

        if step % len(loader_train) == 0:
            epoch = int(step / len(loader_train))
            total_epochs = opt.epochs
            if epoch < int(total_epochs * 0.4):
                eval_every = 10
            elif epoch < int(total_epochs * 0.5):
                eval_every = 10
            elif epoch < int(total_epochs * 0.9):
                eval_every = 10
            else:
                eval_every = 1
            if epoch % eval_every != 0:
                continue
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)

            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            ssim_log.append(ssim_eval)
            eval_epochs.append(epoch)
            plot_psnr_log(psnr_log, epoch, opt.saved_plot_dir, axis_epochs=eval_epochs)
            plot_ssim_log(ssim_log, epoch, opt.saved_plot_dir, axis_epochs=eval_epochs)

            if ssim_eval > max_ssim:
                max_ssim = max(max_ssim, ssim_eval)
            if psnr_eval > max_psnr:
                max_psnr = max(psnr_eval, max_psnr)
                print(
                    f'\n model saved at step :{step}| epoch:{epoch} | max_psnr:{max_psnr:.4f} | ssim:{ssim_eval:.4f}'
                )
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best_model.pk')
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'max_psnr': max_psnr,
                    'psnr_eval': psnr_eval,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, saved_best_model_path)

            saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch)+ '.pk')
            torch.save({
                'epoch': epoch,
                'step': step,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, saved_single_model_path)

            loader = iter(loader_train)
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)


def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache() 
    ssims = []
    psnrs = []
    for i, (inputs, labels, names) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            # inputs = pad_img(inputs,16)
            out, out_grad = net(inputs)
            out = out.clamp(0, 1)
        psnr_tmp = psnr(out, labels).item()
        ssim_tmp = ssim(out, labels).item()
        psnrs.append(psnr_tmp)
        ssims.append(ssim_tmp)
    return np.mean(ssims), np.mean(psnrs)


if __name__ == '__main__':
    set_seed_torch(42)
    train_set = TrainDataset(opt.train_snow_dir, opt.train_gt_dir)
    test_set = TestDataset(opt.val_snow_dir, opt.val_gt_dir, max_images=1000, crop_size=None)

    loader_train = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    loader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    net = fade(channel=40)
    net = net.to(opt.device)

    epoch_size = len(loader_train)
    print("epoch_size", epoch_size)

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(ablation=False))
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(nn.L1Loss().to(opt.device))

    optimizer = torch.optim.Adam(params=filter(lambda x :x.requires_grad, net.parameters()), lr=opt.start_lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion)
