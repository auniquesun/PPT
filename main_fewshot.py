'''
 * Adapted from ULIP (https://github.com/salesforce/ULIP)
 * By Hongyu Sun
'''
from collections import OrderedDict
import os
import math
import time
import wandb

import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import collections

from data.dataset_3d import *

from utils.utils import get_dataset, accuracy, set_random_seed, AverageMeter, ProgressMeter
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils


def main(args):
    utils.init_distributed_mode(args)

    if utils.is_main_process() and args.wandb:
        os.environ["WANDB_BASE_URL"] = args.wb_url
        wandb.login(key=args.wb_key)
        wandb.init(project=args.proj_name, name=args.exp_name)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_random_seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args)
    model.cuda(args.gpu)

    if args.distributed:
        # `find_unused_parameters=False`
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    criterion = CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)

    cudnn.benchmark = True

    # Data loading code
    print(f"=> creating a {args.nshots}-Shot dataset")
    tokenizer = SimpleTokenizer()

    # do not use `train_transform`
    train_dataset = get_dataset(None, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'test')
    print('------ len(train_dataset)', len(train_dataset))
    print('------ len(val_dataset)', len(val_dataset))

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    # --- option 1
    lr_scheduler = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)
    # --- option 2
    # lr_scheduler = utils.cosine_annealing_warmup(optimizer, first_cycle_epochs=args.first_cycle_epochs, 
                # max_lr=args.lr, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, gamma=args.gamma)

    print(args)

    print("=> beginning finetuning")

    best_acc = 0
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, args)
        val_stats = {"acc": -1}

        # lr_scheduler.step()

        val_stats = validate(val_loader, model, criterion, args)
        acc = val_stats["acc"]
        print(val_stats)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if args.distributed:
            state_dict_prompt = model.module.prompt_learner.state_dict()
            # last transformer block
            state_dict_block = model.module.point_encoder.blocks.blocks[-1].state_dict()
        else:
            state_dict_prompt = model.prompt_learner.state_dict()
            # last transformer block
            state_dict_block = model.point_encoder.blocks.blocks[-1].state_dict()

        if is_best:
            best_epoch = epoch
            print("=> saving best checkpoint")
            saved_data = {'epoch': epoch + 1, 'state_dict': state_dict_prompt, # save `prompt_learner` only
                          'optimizer' : optimizer.state_dict(),
                          'best_acc': best_acc,  'args': args,}
            saved_data['last_block'] = state_dict_block if args.head_type > 0 else None

            utils.save_on_master(saved_data, is_best, os.path.join(args.output_dir, args.proj_name, args.exp_name))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc': best_acc,
                     'best_epoch': best_epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
                
    if utils.is_main_process():
        wandb.finish()
        # copy log file from pueue to outputs/${proj_name}/${exp_name}
        utils.copy_log_from_pueue(args.output_dir, args.proj_name, args.exp_name, 'run.log')


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names()
    iters_per_epoch = len(train_loader) // args.update_freq
    # loss & acc
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        if data_iter > len(train_loader) * args.data_ratio: # dr: data_ratio
            break

        optimizer.zero_grad()

        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for _, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_scheduler[it]

        # pc: [batch, npoints, 3]
        pc = inputs[0]
        pc = pc.to(args.gpu)
        # label: [batch, ]
        label = inputs[1].long().to(args.gpu)

        # [batch, num_classes]
        pred = model(pc)
        loss = criterion(pred, label)

        loss.backward(retain_graph=True)
        optimizer.step()

        pred_idx = pred.argmax(dim=1)
        correct = torch.eq(pred_idx, label).sum()
        acc = correct / len(label)

        # NOTE check whether `loss` is exploded
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # clamp logit scale to [0, 100]
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        metrics['loss'].update(loss.item())
        metrics['acc'].update(acc.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            # if utils.is_main_process() and args.wandb:
            #     wandb.log({**{'loss': loss.item(), 'acc': acc.item()},
            #             'logit': logit_scale})
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    val_top1 = AverageMeter('Acc@1', ':6.2f')
    val_loss = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, val_top1, val_loss],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)

        for i, inputs in enumerate(test_loader):
            pc = inputs[0]  # [batch, npoints, 3]
            target = inputs[1]  # [batch,]
            target_name = inputs[2] # [batch]

            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu)
            target = target.long().cuda(args.gpu)

            # [batch, num_classes]
            pred = model(pc)
            loss = criterion(pred, target)

            # compute top1 only
            res, correct = accuracy(pred, target, topk=(1,))
            acc = res[0]

            val_loss.update(loss.item(), pc.size(0))
            val_top1.update(acc.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # [batch]
            top1_accurate = correct[:1].squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.values()]))

    progress.synchronize()
    print('Test * Acc@1 {top1.avg:.3f} Loss {val_loss.avg:.3f}')
    return {'acc': val_top1.avg, 'loss': val_loss.avg}


if __name__ == '__main__':
    from parser import args

    main(args)
