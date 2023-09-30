'''
 * Adapted from ULIP (https://github.com/salesforce/ULIP)
 * By Hongyu Sun
'''
from collections import OrderedDict
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

from data.dataset_3d import *

from utils.utils import get_dataset, to_categorical, set_random_seed, AverageMeter, ProgressMeter
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
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

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
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()

    # do not use `train_transform`
    train_dataset = get_dataset(None, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'val')
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

    best_test_acc = 0
    best_mean_class_iou = 0
    best_mean_inst_iou = 0
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, args)
        val_stats = validate(val_loader, model, criterion, args)
        # lr_scheduler.step()

        test_acc = val_stats["acc"]
        test_mean_class_iou = val_stats["mean_class_iou"]
        test_mean_inst_iou = val_stats["mean_inst_iou"]
        print(val_stats)

        is_best = test_mean_class_iou > best_mean_class_iou
        best_test_acc = max(test_acc, best_test_acc)
        best_mean_class_iou = max(test_mean_class_iou, best_mean_class_iou)
        best_mean_inst_iou = max(test_mean_inst_iou, best_mean_inst_iou)

        if is_best:
            best_epoch = epoch
            print(f"=> find best Mean Class IoU `{best_mean_class_iou}` at Epoch `{epoch}`")
            print("=> saving best checkpoint")

            if args.distributed:
                state_dict_prompt = model.module.prompt_learner.state_dict()
                state_dict_partseg = model.module.point_encoder.state_dict()
            else:
                state_dict_prompt = model.prompt_learner.state_dict()
                state_dict_partseg = model.point_encoder.state_dict()

            utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict_prompt': state_dict_prompt, # save `prompt_learner`
                    'state_dict_partseg': state_dict_partseg, # save `point_encoder`
                    'optimizer' : optimizer.state_dict(),
                    'best_test_acc': best_test_acc,
                    'best_mean_class_iou': best_mean_class_iou,
                    'best_mean_inst_iou': best_mean_inst_iou,
                    'args': args,
                }, is_best, os.path.join(args.output_dir, args.proj_name, args.exp_name))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'best_test_acc': best_test_acc, 
                     'best_mean_inst_iou': best_mean_inst_iou,
                     'test_mean_inst_iou': test_mean_inst_iou,
                     'best_mean_class_iou': best_mean_class_iou,
                     'test_mean_class_iou': test_mean_class_iou,
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
    
    part2category = train_loader.dataset.part2category
    category2part = train_loader.dataset.category2part
    num_part_classes = len(part2category)
    num_shape_classes = len(category2part)

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        if data_iter > len(train_loader) * args.data_ratio:
            break

        optimizer.zero_grad()

        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_scheduler[it]

        # pc: [batch, npoints, 3]
        pc = inputs[0].to(args.gpu)
        # cls_label: [batch,]
        cls_label = inputs[1].to(args.gpu)
        # part_label: [batch, npoints]
        part_label = inputs[2].long().to(args.gpu)

        # [batch, npoints, num_part_classes]
        part_pred = model(pc, to_categorical(cls_label, num_shape_classes, args)) # there are `16` shape classes in shapenetpart
        
        # NOTE it is neccessay to reshape `part_pred` and `part_label` to use CrossEntropyLoss
        loss = criterion(part_pred.reshape(-1, num_part_classes), part_label.reshape(-1))

        loss.backward(retain_graph=True)
        optimizer.step()

        batch_size, npoints, _ = pc.size()
        refined_part_pred = torch.zeros(batch_size, npoints, dtype=torch.int32, device=f'cuda:{args.gpu}')
        for i in range(batch_size):
            # part_label[i, 0] is a tensor, it should be converted to an integer to serve as an index
            idx = part_label[i, 0].item()
            cat = part2category[idx]
            logits = part_pred[i, :, :]
            refined_part_pred[i, :] = torch.argmax(logits[:, category2part[cat]], dim=1) + category2part[cat][0]

        # compute overall accuracy
        correct = torch.eq(refined_part_pred, part_label).sum()
        acc = correct / (batch_size*npoints)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # clamp logit scale to [0, 100]
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        metrics['loss'].update(loss.item(), n=batch_size)
        metrics['acc'].update(acc.item(), n=batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


@torch.no_grad()
def validate(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    val_top1 = AverageMeter('Acc@1', ':6.2f')
    val_loss = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, val_top1, val_loss],
        prefix='Test: ')
    
    part2category = test_loader.dataset.part2category
    category2part = test_loader.dataset.category2part
    num_shape_classes = len(category2part)
    num_part_classes = len(part2category)

    part2correct = torch.zeros(num_part_classes, device=f'cuda:{args.gpu}')
    part2total = torch.zeros(num_part_classes, device=f'cuda:{args.gpu}')
    inst_ious = []
    shape_ious = {obj:[] for obj in category2part.keys()}

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, inputs in enumerate(test_loader):
        # [batch, npoints, 3]
        pc = inputs[0].to(args.gpu)
        # [batch,]
        cls_label = inputs[1].to(args.gpu)
        # part_label: [batch, npoints]
        part_label = inputs[2].long().to(args.gpu)

        # [batch, num_classes]
        part_pred = model(pc, to_categorical(cls_label, num_shape_classes, args)) # there are `16` shape classes in shapenetpart
        loss = criterion(part_pred.reshape(-1, num_part_classes), part_label.reshape(-1))

        # NOTE: the following loop ensures the predictions happen on the target object category
        batch_size, npoints, _ = pc.size()
        refined_part_pred = torch.zeros(batch_size, npoints, dtype=torch.int32, device=f'cuda:{args.gpu}')
        for i in range(batch_size):
            # partseg_label[i, 0] is a tensor, it should be converted to an integer to serve as an index
            idx = part_label[i, 0].item()
            cat = part2category[idx]
            logits = part_pred[i, :, :]
            refined_part_pred[i, :] = torch.argmax(logits[:, category2part[cat]], dim=1) + category2part[cat][0]

        correct = torch.eq(refined_part_pred, part_label).sum()
        acc = correct / (batch_size*npoints)

        val_top1.update(acc.item(), pc.size(0))
        val_loss.update(loss.item(), pc.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for i in range(num_part_classes):
            # torch.eq expects one of 
            #   * (Tensor input, Tensor other, *, Tensor out)
            #   * (Tensor input, Number other, *, Tensor out)
            part2correct[i] += torch.eq(refined_part_pred, i).sum()
            part2total[i] += torch.eq(part_label, i).sum()

        for i in range(batch_size):
            pred, gt = refined_part_pred[i, :], part_label[i, :]
            part = gt[0].item()    # `0th` point in the point cloud belongs to one part, gt[0].item() converts to a number to serve as an index
            cat = part2category[part]     # find the object category of this part
            intersection = torch.eq(gt, pred).sum()
            union = len(gt)
            inst_iou = intersection / union
            inst_ious.append(inst_iou)

            # compute for all parts of a `single` object
            part_ious = [.0 for _ in range(len(category2part[cat]))]
            for j, part in enumerate(category2part[cat]):
                if j == 0:
                    start_id = category2part[cat][0]
                if torch.logical_or(torch.eq(gt, part), torch.eq(pred, part)).sum() == 0:
                    part_ious[part - start_id] = 1
                else:
                    intersection = torch.logical_and(torch.eq(gt, part), torch.eq(pred, part)).sum()
                    union = torch.logical_or(torch.eq(gt, part), torch.eq(pred, part)).sum()
                    part_ious[part - start_id] = intersection / union
            shape_ious[cat].append(torch.mean(torch.tensor(part_ious)))

        if i % args.print_freq == 0:
            progress.display(i)

    all_inst_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_inst_ious.append(iou)
        shape_ious[cat] = torch.mean(torch.tensor(shape_ious[cat]))
        print('Category:', cat, ' ||  Category IoU:', shape_ious[cat])
    
    mean_inst_iou = torch.mean(torch.tensor(all_inst_ious))
    category_ious = [cat_iou for cat_iou in shape_ious.values()]
    mean_class_iou = torch.mean(torch.tensor(category_ious))

    progress.synchronize()
    print(f'Test * Acc@1 : {val_top1.avg:.3f} Loss : {val_loss.avg:.3f} '
          f'Mean Instance IoU : {mean_inst_iou.item()} '
          f'Mean Category IoU : {mean_class_iou.item()}')
    
    return {'acc': val_top1.avg, 'loss': val_loss.avg, 
            'mean_inst_iou': mean_inst_iou.item(), 
            'mean_class_iou': mean_class_iou.item()}


if __name__ == '__main__':
    from parser import args

    main(args)
