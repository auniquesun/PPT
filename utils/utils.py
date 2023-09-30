import os
import json
import random
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
import torch.autograd as autograd

from PIL import ImageFilter
from easydict import EasyDict
import yaml
from data.dataset_3d import Dataset_3D

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        # try:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
        # except:
        #     new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, is_best, save_dir):
    if is_main_process():
        best_path = f'{save_dir}/checkpoint_best.pt'
        if is_best: # only save best checkpoint
            torch.save(state, best_path)


def init_distributed_mode(args):
    dst_dir = os.path.join(args.output_dir, args.proj_name, args.exp_name)
    cpy_dir = os.path.join(dst_dir, 'copy')
    script = os.path.join('scripts', args.proj_name, f'{args.exp_name}.sh')
    # create all intermediate-level directories
    os.makedirs(cpy_dir, exist_ok=True)

    shutil.copy(args.main_program, cpy_dir)
    shutil.copy(script, cpy_dir)
    shutil.copy('parser.py', cpy_dir)
    shutil.copy('utils/utils.py', cpy_dir)
    shutil.copy('data/dataset_3d.py', cpy_dir)
    shutil.copy('models/ULIP_models.py', cpy_dir)

    with open('data/labels.json') as fin:
        labels = json.load(fin)
        args.classnames = labels[args.dataset_name]

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.

    tensors: [pc_embed, text_embed, image_embed]
                pc_embed: [batch, embed_dim]
                text_embed/image_embed 和 pc_embed 维度一样
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        # 我记得这个操作是相当消耗显存的，要把多卡的 point, text, image 特征聚集到一起
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        # `tensor_all` is a list, e.g. [tensor_1, ..., tensor_i, ..., tensor_n], among them, tensor_i: [batch, embed_dim]
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        # torch.cat(tensor_all, dim=0)  -> [world_size*batch, embed_dim]
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def cosine_annealing_warmup(optimizer, first_cycle_epochs, max_lr, min_lr, warmup_epochs, gamma=0.5):
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_epochs,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_epochs,
        gamma=gamma
    )
    return lr_scheduler    


def copy_log_from_pueue(output_dir, proj_name, exp_name, log_file):
    pueue_log_dir = '/home/jerry/.local/share/pueue/task_logs'
    dst_dir = os.path.join(output_dir, proj_name, exp_name)
    file = os.path.join(dst_dir, log_file)
    with open(file) as fin:
        line = fin.readline()   # read only one line

        start_idx = line.find('id') + 3
        end_idx = line.find(')')

        task_id = int(line[start_idx: end_idx])
        print(f'task ID: {task_id}')

        try:
            src_file = os.path.join(pueue_log_dir, f'{task_id}.log')
            end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dst_file = os.path.join(dst_dir, f'[{task_id}]_{end_time}.log')
            shutil.copy(src_file, dst_file)
            print(f'--- Successfully copy `{task_id}.log` ---')
        except Exception as e:
            print(f'=== Failed. There are something wrong! ===')
            print(e)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_dataset(train_transform, tokenizer, args, dataset_type=None):
    dataset_3d = Dataset_3D(args, tokenizer, dataset_type, train_transform)
    return dataset_3d.dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
        output: [batch, num_labels]
        target: [batch,]
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # pred: [batch, maxk], storing the indices of topk elements in each row
        _, pred = output.topk(maxk, 1, True, True)
        # pred: [maxk, batch]
        pred = pred.t()
        # target.reshape(1, -1)                 -> [1, batch]
        # target.reshape(1, -1).expand_as(pred) -> [maxk, batch]
        # correct:                              -> [maxk, batch]
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res, correct


def to_categorical(label, num_classes, args):
    """
    Args:
        label: [batch]
        num_classes: an integer
    Return:
        new_label: [batch, num_classes]
    """
    new_label = torch.eye(num_classes)[label.cpu().numpy(),]
    if label.is_cuda:
        return new_label.to(args.gpu)
    return new_label