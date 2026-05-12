import os

from matplotlib import pyplot as plt
import numpy as np

def plot_loss_log(loss_log, epoch, loss_dir):
    axis = np.linspace(1, epoch, epoch)
    for key in loss_log.keys():
        label = '{} Loss'.format(key)
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, np.array(loss_log[key]), label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(loss_dir, 'loss_{}.pdf'.format(key)))
        plt.close(fig)

def plot_psnr_log(psnr_log, epoch, psnr_dir, axis_epochs=None):
    if axis_epochs is not None and len(axis_epochs) == len(psnr_log):
        axis = np.array(axis_epochs)
    else:
        axis = np.arange(1, len(psnr_log) + 1)
    label = 'PSNR'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(psnr_log), label=label)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(psnr_dir, 'psnr.pdf'))
    plt.close(fig)

def plot_ssim_log(ssim_log, epoch, ssim_dir, axis_epochs=None):
    if axis_epochs is not None and len(axis_epochs) == len(ssim_log):
        axis = np.array(axis_epochs)
    else:
        axis = np.arange(1, len(ssim_log) + 1)
    label = 'SSIM'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(ssim_log), label=label)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.savefig(os.path.join(ssim_dir, 'ssim.pdf'))
    plt.close(fig)

