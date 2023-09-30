import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from data.dataset_3d import *

from utils.utils import set_random_seed, get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils


def main(args):
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_random_seed(seed)

    # --- 1.1 define model
    print("=> creating model: {}".format(args.model))
    with open('data/labels.json') as fin:
        labels = json.load(fin)
        args.classnames = labels[args.dataset_name]
    model = getattr(models, args.model)(args)
    model.cuda(args.gpu)

    # --- 1.2 load model weights
    if args.head_type > 0 and 'pointbert' in args.model.lower():
        weights_path = os.path.join('outputs', args.proj_name, args.exp_name, 'checkpoint_best.pt')
        state_dict = torch.load(weights_path)
        # last transformer block in pointbert
        state_dict['last_block'] = {'point_encoder.blocks.blocks.11.' + key: value for key, value in state_dict['last_block'].items()}    
        print('=> loading the weights of last transformer block')
        model.load_state_dict(state_dict['last_block'], strict=False)

    # --- 2. load data
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()

    # do not use `train_transform`
    dataset = get_dataset(None, tokenizer, args, 'test')
    print('------ len(dataset)', len(dataset))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    print(args)
    print(f"=> extracting feature for {args.dataset_name} test set")

    # switch to evaluate mode
    model.eval()

    # --- 3. forward pass
    with torch.no_grad():
        test_feats = []
        test_labels = []
        test_names = []
        for _, inputs in enumerate(data_loader):
            pc = inputs[0]  # [batch, npoints, 3]
            target = inputs[1]  # [batch,]
            target_name = inputs[2] # [batch]

            test_labels.extend(target.tolist())
            test_names.extend(target_name)

            pc = pc.cuda(args.gpu)
            target = target.long().cuda(args.gpu)

            # [batch, num_classes]
            feats = model(pc)
            test_feats.extend(feats.tolist())

        state_dict = {'test_feats': np.array(test_feats), 
                      'test_labels': np.array(test_labels),
                      'test_names': np.array(test_names)}
        torch.save(state_dict, f'notebook/{args.dataset_name}_test_feats_labels.pt')
        print('=> saved test feats!')


if __name__ == '__main__':
    from parser import args
    # 1. fro mn40, use weights from: inlab_ubuntu/ulip2_data_effi, cls-pointbert-mn40-32v-middle-dr02-1
    # 2. for sonn, use weights from: lyp_ubuntu/ulip2_data_effi_2, cls-pointbert-sonn_hardest-32v-middle-dr05-h3-1
    main(args)
