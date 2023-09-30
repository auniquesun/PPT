import os
import json
import shutil

import torch

from models.ULIP_models import ULIP_PN_SSG, ULIP_PN_MSG, ULIP_PointBERT, ULIP_PointBERT_partseg
from utils.tokenizer import SimpleTokenizer
from utils.utils import get_dataset


if '__main__' == __name__:
    from parser import args

    with open('data/labels.json') as fin:
        labels = json.load(fin)
        args.classnames = labels[args.dataset_name]
   
    # --- pointnet2_ssg
    model = ULIP_PN_SSG(args)

    # --- pointnet2_msg
    model = ULIP_PN_MSG(args)

    # --- pointbert
    model = ULIP_PointBERT(args)

    # --- test how to use last point transformer block
    blocks = model.point_encoder.blocks.blocks
    print('len(blocks):', len(blocks))
    last_block = blocks[-1]
    print('last_block:', last_block)
    print('last_block.state_dict().keys():', last_block.state_dict().keys())

    # --- pointbert_partseg
    model = ULIP_PointBERT_partseg(args)

    # --- get the dir of current file 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print('BASE_DIR:', BASE_DIR)

    PROJ_DIR = os.path.dirname(BASE_DIR)
    print('PROJ_DIR:', PROJ_DIR)

    # --- count learnable parameters
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count += 1
            print('======:', name)
    print(f'>>>>>> {count}')

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', params)

    # --- dataset
    tokenizer = SimpleTokenizer()

    train_dataset = get_dataset(None, tokenizer, args, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    # --- access an `attribute` in a Dataset
    print(train_loader.dataset.root)
    
    print('--- len(train_dataset):', len(train_dataset))
    print('--- len(train_loader):', len(train_loader))

    print('args.data_ratio:', args.data_ratio)
    print('args.data_ratio:', args.data_ratio)
    print(args)

    # --- replace `atten_head` with `mlp_head` in `scripts/ulip2_head_type_fewshot`
    base_dir = 'scripts/ulip2_head_type_fewshot'

    for f in os.listdir(base_dir):
        f_new = f.replace('atten_head', 'mlp_head')
        shutil.move(os.path.join(base_dir, f), os.path.join(base_dir, f_new))
        print(f'>>> rename to [{f_new}]')

    # --- replace `mlp_head` with `lin_head` in `scripts/ulip2_head_type_lin_fs`
    base_dir = 'scripts/ulip2_head_type_lin_fs'

    for f in os.listdir(base_dir):
        f_new = f.replace('mlp_head', 'lin_head')
        shutil.move(os.path.join(base_dir, f), os.path.join(base_dir, f_new))
        print(f'>>> rename to [{f_new}]')
