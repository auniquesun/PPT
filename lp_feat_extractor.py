import os
import numpy as np
import torch

from utils.utils import get_dataset, init_distributed_mode, set_random_seed, copy_log_from_pueue
from utils.tokenizer import SimpleTokenizer
from data.dataset_3d import *

import models.ULIP_models as models


def main(args):
    init_distributed_mode(args)

    if args.seed >= 0:
        seed = args.seed
        set_random_seed(seed)
        print("Setting fixed seed: {}".format(seed))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    ######################################
    #   Setup DataLoader
    ######################################
    tokenizer = SimpleTokenizer()

    if args.dataset_type == 'train':
        dataset = get_dataset(None, tokenizer, args, 'train')
        shuffle = True
        print('------ len(train_dataset)', len(dataset))
    else:
        dataset = get_dataset(None, tokenizer, args, 'test')
        shuffle = False
        print('------ len(val_dataset)', len(dataset))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, 
        num_workers=args.workers, pin_memory=True, drop_last=False)

    ########################################
    #   Setup Network
    ########################################
    model = getattr(models, args.model)(args)
    point_encoder = model.point_encoder.to(args.gpu)
    point_encoder.eval()

    ###################################################################################################################
    # Start Feature Extractor
    feature_list = []
    label_list = []
    
    for _, inputs in enumerate(data_loader):
        pc = inputs[0].to(args.gpu)
        feature = point_encoder(pc)
        feature = feature.cpu()

        for idx in range(len(pc)):
            feature_list.append(feature[idx].tolist())
        label_list.extend(inputs[1].tolist())

    save_dir = os.path.join(args.output_dir, args.proj_name, args.exp_name)
    
    np.savez(
        os.path.join(save_dir, args.dataset_type),
        feature_list=feature_list,
        label_list=label_list,
    )

    copy_log_from_pueue(args.output_dir, args.proj_name, args.exp_name, 'run.log')


if __name__ == "__main__":
    from parser import args
    main(args)
