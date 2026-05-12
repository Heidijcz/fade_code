import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default=r'../experiments/test')
parser.add_argument('--dataset', type=str, default='snow100k')
parser.add_argument('--val_snow_dir', type=str, default=r"..\datasets\Snow100K\realistic")
parser.add_argument('--val_gt_dir', type=str, default=r"..\datasets\Snow100K_small\test\Snow100K-L\gt")

parser.add_argument('--model_name', type=str, default='fade')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

# only need for evaluation
parser.add_argument('--pre_trained_model', type=str, default=r"..\checkpoint\snow100k_best_model.pk", help='path of pre trained model for resume training')
parser.add_argument('--save_infer_results', action='store_true', default=False, help='save the infer results during validation')
opt = parser.parse_args()

opt.different_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
opt.different_dataset_model_dir = os.path.join(opt.different_dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.makedirs(opt.exp_dir)
if not os.path.exists(opt.different_dataset_dir):
    os.mkdir(opt.different_dataset_dir)
if not os.path.exists(opt.different_dataset_model_dir):
    os.mkdir(opt.different_dataset_model_dir)

pre_trained_model_name = opt.pre_trained_model.split('\\')[-1]
opt.saved_infer_dir = os.path.join(opt.different_dataset_model_dir, pre_trained_model_name.split('.pk')[0])
if not os.path.exists(opt.saved_infer_dir):
    os.mkdir(opt.saved_infer_dir)

with open(os.path.join(opt.different_dataset_model_dir, 'opt.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
