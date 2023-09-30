import os
import argparse
from dotenv import load_dotenv


parser = argparse.ArgumentParser(description='PPT training and evaluation', add_help=False)
# Data
parser.add_argument('--output_dir', default='outputs', type=str, help='output dir')
parser.add_argument('--dataset_name', default='modelnet40', type=str)
parser.add_argument('--dataset_type', default='test', type=str, choices=['train', 'val', 'test'])
parser.add_argument('--sonn_type', default='hardest', type=str, choices=['obj_only', 'obj_bg', 'hardest'])
parser.add_argument('--dataset_prompt', default='modelnet40_64', type=str)
parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
parser.add_argument('--nshots', default=16, type=int, help='number of shot for each category in train set.')
# Prompt
parser.add_argument('--template_init', default='', type=str)
parser.add_argument('--num_learnable_prompt_tokens', default=32, type=int)
parser.add_argument('--class_name_position', default='end', type=str)
# Model
parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
parser.add_argument("--head_type", type=int, default=0, choices=[0, 1, 2, 3], 
                    help="0: prompt_only, 1: linear, 2: mlp, 3: atten_block")
parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
parser.add_argument('--ulip2', action='store_true', help='use the pretrained model of ULIP2')
parser.add_argument("--fpath", type=str, help="Path to the learned prompt")
parser.add_argument("--topk", type=int, help="Select top-k similar words")
# Training
parser.add_argument('--epochs', default=250, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int,
                    help='number of samples per-device/per-gpu')
parser.add_argument('--data_ratio', default=1.0, type=float, help='used data ratio of training set')
parser.add_argument('--optim', default='adamw', type=str)
parser.add_argument('--first_cycle_epochs', default=5, type=int, help='a parameter in cosine annealing warmup restart')
parser.add_argument('--lr', default=3e-3, type=float, help='initial learning rate')
parser.add_argument('--max_lr', default=3e-3, type=float, help='max_lr in cosine annealing warmup restart')
parser.add_argument('--min_lr', default=.0, type=float, help='min_lr in cosine annealing warmup restart')
parser.add_argument('--gamma', default=0.5, type=float, help='gamma in cosine annealing warmup restart')
parser.add_argument('--lr_start', default=1e-6, type=float, help='initial warmup lr')
parser.add_argument('--lr_end', default=1e-5, type=float, help='minimum final lr')
parser.add_argument('--update_freq', default=1, type=int,
                    help='optimizer update frequency (i.e. gradient accumulation steps)')
parser.add_argument('--wd', default=0.1, type=float)
parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--disable-amp', action='store_true',
                    help='disable mixed-precision training (requires more memory and compute)')
parser.add_argument('--resume', default='', type=str, help='path to resume from')
parser.add_argument('--label_smoothing', default=0.3, type=float, help='label smoothing in cross entropy loss')
# Linear probe
parser.add_argument("--num_step", type=int, default=8, help="number of steps to conduct binary search")
parser.add_argument("--num_run", type=int, default=10, help="number of runs for a n-shot linear experiment")
# System
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers per process')
parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--task', default='cls', type=str, choices=['cls', 'fewshot', 'partseg', 'fs_lp'], help='task type.')
# wandb
parser.add_argument('--proj_name', type=str, default="L2P3D", help='project name')
parser.add_argument('--exp_name', type=str, default="", help='experiment name')
parser.add_argument('--main_program', type=str, default="", help='main program for target task')
parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
parser.add_argument('--wb_url', type=str, default="", help='wandb server url')
parser.add_argument('--wb_key', type=str, default="", help='wandb login key')

# load `wb_url` and `wb_key` from environment variables, rather than using `explicit strings`
load_dotenv()
args = parser.parse_args()
args.wb_url = os.environ['wb_url']
args.wb_key = os.environ['wb_key']
