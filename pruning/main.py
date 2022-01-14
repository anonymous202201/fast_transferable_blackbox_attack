import argparse
import os
import random
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import warnings
import yaml

import admm
from utils import calculate_var_reg
from utils import get_layer
from utils import get_pruning_names

import pdb


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')

parser.add_argument(
    '--load_model',
    type=str,
    default="pretrained_mnist.pt",
    help='For loading the model')
parser.add_argument(
    '--masked_retrain',
    action='store_true',
    default=False,
    help='for masked retrain')
parser.add_argument(
    '--verbose',
    action='store_true',
    default=False,
    help='whether to report admm convergence condition')
parser.add_argument(
    '--admm',
    action='store_true',
    default=False,
    help="for admm training")
parser.add_argument(
    '--admm_conf_path', type=str,
    default="config.yaml",
    help="Config path for ADMM pruning.")
parser.add_argument('--admm_epoch', type=int, default=1,
                    help="how often we do admm update")
parser.add_argument(
    '--rho',
    type=float,
    default=0.001,
    help="define rho for ADMM")
parser.add_argument(
    '--var_reg_weight',
    type=float,
    default=1e-4,
    help="define weight for variance regularization.")
parser.add_argument(
    '--sparsity_type',
    type=str,
    default='irregular',
    help="define sparsity_type: [irregular,column,filter]")


best_acc1 = 0
GLOBAL_STEPS=0


def main():
  args = parser.parse_args()

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                  'disable data parallelism.')

  if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(
        main_worker,
        nprocs=ngpus_per_node,
        args=(
            ngpus_per_node,
            args))
  else:
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
  global best_acc1
  global GLOBAL_STEPS
  args.gpu = gpu
  writer = SummaryWriter()

  if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

  if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
      args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
      # For multiprocessing distributed training, rank needs to be the
      # global rank among all the processes
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
  # create model
  if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.arch))
    if args.arch == "alexnet":
      raise NameError("not implemented yet")
    model = models.__dict__[args.arch](pretrained=True)
  else:
    print("=> creating model '{}'".format(args.arch))
    if args.arch == "alexnet":
      model = AlexNet_BN()
      print (model)
      for i, (name, W) in enumerate(model.named_parameters()):
          print (name)
    else:
      model = models.__dict__[args.arch]()

  if args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)
      model.cuda(args.gpu)
      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs we have
      args.batch_size = int(args.batch_size / ngpus_per_node)
      args.workers = int(args.workers / ngpus_per_node)
      model = torch.nn.parallel.DistributedDataParallel(
          model, device_ids=[args.gpu])
    else:
      model.cuda()
      # DistributedDataParallel will divide and allocate batch_size to all
      # available GPUs if device_ids are not set
      model = torch.nn.parallel.DistributedDataParallel(model)
  elif args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
  else:
    # DataParallel will divide and allocate batch_size to all available
    # GPUs
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
      model.features = torch.nn.DataParallel(model.features)
      model.cuda()
    else:
      model = torch.nn.DataParallel(model).cuda()

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)

  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      GLOBAL_STEPS = checkpoint['GLOBAL_STEPS']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  traindir = os.path.join(args.data, 'train')
  valdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
  else:
    train_sampler = None

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=(
          train_sampler is None),
      num_workers=args.workers,
      pin_memory=True,
      sampler=train_sampler)

  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ])),
      batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)

  if args.evaluate:
    validate(val_loader, model, criterion, args)
    return
  ADMM = admm.ADMM(model, args.admm_conf_path)

  pruning_names = get_pruning_names(args.admm_conf_path)

  admm.admm_initialization(
      args,
      ADMM=ADMM,
      model=model)  # intialize Z variable

  for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
      train_sampler.set_epoch(epoch)
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train(train_loader, ADMM, model, criterion, optimizer,
          epoch, args, writer, pruning_names)

    # evaluate on validation set
    acc1, acc5 = validate(val_loader, model, criterion, args)
    writer.add_scalar("evaluate/acc1", acc1, epoch)
    writer.add_scalar("evaluate/acc5", acc5, epoch)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank %
            ngpus_per_node == 0):
      save_checkpoint(args, {
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': model.state_dict(),
          'best_acc1': best_acc1,
          'GLOBAL_STEPS': GLOBAL_STEPS,
          'optimizer': optimizer.state_dict(),
      }, is_best)


def train(train_loader, ADMM, model, criterion, optimizer,
          epoch, args, writer, pruning_names):
  global GLOBAL_STEPS
  batch_time = AverageMeter()
  data_time = AverageMeter()
  ce_losses = AverageMeter()
  mixed_losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  variance_reges = AverageMeter()
  admm_losses = {}
  with open(args.admm_conf_path, 'r') as f:
    raw_dict = yaml.load(f)
    for k in raw_dict['prune_ratios'].keys():
      admm_losses[k] = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  handles = []
  for step_idx, (input, target) in enumerate(train_loader):
    GLOBAL_STEPS += 1
    # measure data loading time
    data_time.update(time.time() - end)

    if args.gpu is not None:
      input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)

    # compute output
    features = []
    if step_idx == 0:
      for (name, W) in model.named_parameters():
        if name not in pruning_names:
          continue
        def hook(model, input, output):
          features.append(output.clone())
        handle = get_layer(model, name, args.arch).register_forward_hook(hook)
        handles.append(handle)
    output = model(input)
    ce_loss = criterion(output, target)
    var_reg = calculate_var_reg(features)

    admm.z_u_update(
        args,
        ADMM,
        model,
        None,
        train_loader,
        optimizer,
        epoch,
        input,
        step_idx)   # update Z and U variables

    ret_dict = admm.append_admm_loss(
        args, ADMM, model, ce_loss, var_reg)  # append admm losss
    ce_loss = ret_dict["ce_loss"]
    admm_loss = ret_dict["admm_loss"]
    mixed_loss = ret_dict["mixed_loss"]
    variance_reg = ret_dict["variance_reg"]
    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    ce_losses.update(ce_loss.item(), input.size(0))
    for k, v in admm_loss.items():
      admm_losses[k].update(v.item(), input.size(0))
    variance_reges.update(variance_reg.item(), input.size(0))
    mixed_losses.update(mixed_loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    mixed_loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    writer.add_scalar("epoch", epoch, GLOBAL_STEPS)
    writer.add_scalar("loss/ce_loss", ce_losses.val, GLOBAL_STEPS)
    writer.add_scalar("loss/ce_loss_avg", ce_losses.avg, GLOBAL_STEPS)
    writer.add_scalar("loss/variance_reg", variance_reges.val, GLOBAL_STEPS)
    writer.add_scalar("loss/variance_reg_avg", variance_reges.avg, GLOBAL_STEPS)
    for k, v in admm_losses.items():
      writer.add_scalar("admm_loss/_{}".format(k), v.val, GLOBAL_STEPS)
      writer.add_scalar("admm_loss/_{}_avg".format(k), v.avg, GLOBAL_STEPS)
    writer.add_scalar("loss/mixed_loss", mixed_losses.val, GLOBAL_STEPS)
    writer.add_scalar("loss/mixed_loss_avg", mixed_losses.avg, GLOBAL_STEPS)
    writer.add_scalar("acc/acc1", top1.val, GLOBAL_STEPS)
    writer.add_scalar("acc/acc1_avg", top1.avg, GLOBAL_STEPS)
    writer.add_scalar("acc/acc5", top5.val, GLOBAL_STEPS)
    writer.add_scalar("acc/acc5_avg", top5.avg, GLOBAL_STEPS)

    if step_idx % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, step_idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=ce_losses, top1=top1, top5=top5))
      print (
          "cross_entropy loss: {}, mixed_loss : {}".format(
              ce_loss, mixed_loss))
    if args.verbose:
      for k, v in admm_loss.items():
        print ("at layer {}, admm loss is {}".format(k, v))
  for handle in handles:
    handle.remove()


def validate(val_loader, model, criterion, args):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
      if args.gpu is not None:
        input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(acc1[0], input.size(0))
      top5.update(acc5[0], input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0:
        print(
            'Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i,
                len(val_loader),
                batch_time=batch_time,
                loss=losses,
                top1=top1,
                top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

  return top1.avg, top5.avg


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
  filename = 'checkpoint_gpu{}.pth.tar'.format(args.gpu)
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].contiguous(
      ).view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
  main()
