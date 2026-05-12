import argparse
import os
import json
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--start_lr', type=float, default=0.0003)
parser.add_argument('--end_lr', type=float, default=0.000001)
parser.add_argument('--no_lr_sche', action='store_true',help='no lr cos schedule')
parser.add_argument('--use_warm_up', type=bool, default=False, help='using warm up in learning rate')

parser.add_argument('--w_loss_L1', type=float, default=1.0)
parser.add_argument('--w_loss_CR', type=float, default=0.1)
parser.add_argument('--w_loss_Grad', type=float, default=0.2)
parser.add_argument('--w_loss_Pre_amp', type=float, default=0.05)
parser.add_argument('--w_loss_Pre_pha', type=float, default=0.05)

parser.add_argument('--exp_dir', type=str, default=r'../experiments/train')
parser.add_argument('--model_name', type=str, default='FADE')
parser.add_argument('--saved_model_dir', type=str, default='saved_model')
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
parser.add_argument('--saved_plot_dir', type=str, default='saved_plot')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer')

parser.add_argument('--dataset', type=str, default='Snow100K')
parser.add_argument('--train_snow_dir', type=str, default=r"..\datasets\Snow100K\train\synthetic")
parser.add_argument('--train_gt_dir', type=str, default=r"..\datasets\Snow100K\train\gt")
parser.add_argument('--val_snow_dir', type=str, default=r"..\datasets\Snow100K\test\Snow100K-L\synthetic")
parser.add_argument('--val_gt_dir', type=str, default=r"..\datasets\Snow100K\test\Snow100K-L\gt")

# only need for resume
parser.add_argument('--resume', type=bool,default=False)
parser.add_argument('--pre_trained_model', type=str,default='null')

opt = parser.parse_args()
opt.device = 'cuda'if torch.cuda.is_available() else 'cpu'

# Create save paths for different datasets
different_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
different_dataset_model_dir = os.path.join(different_dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.makedirs(opt.exp_dir)
if not os.path.exists(different_dataset_dir):
    os.makedirs(different_dataset_dir)
if not os.path.exists(different_dataset_model_dir):
    os.makedirs(different_dataset_model_dir)

opt.saved_model_dir = os.path.join(different_dataset_model_dir, 'saved_model')
opt.saved_data_dir = os.path.join(different_dataset_model_dir, 'saved_data')
opt.saved_plot_dir = os.path.join(different_dataset_model_dir, 'saved_plot')
opt.saved_infer_dir = os.path.join(different_dataset_model_dir, 'saved_infer')

if not os.path.exists(opt.saved_model_dir):
    os.mkdir(opt.saved_model_dir)
if not os.path.exists(opt.saved_data_dir):
    os.mkdir(opt.saved_data_dir)
if not os.path.exists(opt.saved_plot_dir):
    os.mkdir(opt.saved_plot_dir)
if not os.path.exists(opt.saved_infer_dir):
    os.mkdir(opt.saved_infer_dir)

with open(os.path.join(different_dataset_model_dir, 'opt.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
