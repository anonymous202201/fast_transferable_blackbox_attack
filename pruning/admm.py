from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from numpy import linalg as LA
import yaml

import pdb


class ADMM:
  def __init__(self, model, file_name, rho=0.001):
    self.ADMM_U = {}
    self.ADMM_Z = {}
    self.rho = rho
    self.rhos = {}

    self.init(file_name, model)

  def init(self, config, model):
    """
    Args:
        config: configuration file that has settings for prune ratios, rhos
    called by ADMM constructor. config should be a .yaml file

    """
    if not isinstance(config, str):
      raise Exception("filename must be a str")
    with open(config, "r") as stream:
      try:
        raw_dict = yaml.load(stream)
        self.prune_ratios = raw_dict['prune_ratios']
        for k, v in self.prune_ratios.items():
          self.rhos[k] = self.rho
        for (name, W) in model.named_parameters():
          if name not in self.prune_ratios:
            continue
          self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
          self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z

      except yaml.YAMLError as exc:
        print(exc)


def random_pruning(args, weight, prune_ratio):
  """
  random_pruning for comparison

  """
  pass


def pgd_pruning(args, weight, prune_ratio):
  """
  projected gradient descent for comparison

  """
  pass


def weight_pruning(args, weight, prune_ratio):
  """
  weight pruning [irregular,column,filter]
  Args:
        weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
        prune_ratio (float between 0-1): target sparsity of weights

  Returns:
        mask for nonzero weights used for retraining
        a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

  """

  weight = weight.cpu().detach().numpy()            # convert cpu tensor to numpy

  percent = prune_ratio * 100
  if (args.sparsity_type == "irregular"):
    # a buffer that holds weights with absolute values
    weight_temp = np.abs(weight)
    # get a value for this percentitle
    percentile = np.percentile(weight_temp, percent)
    under_threshold = weight_temp < percentile
    above_threshold = weight_temp > percentile
    # has to convert bool to float32 for numpy-tensor conversion
    above_threshold = above_threshold.astype(np.float32)
    weight[under_threshold] = 0
    return torch.from_numpy(above_threshold).cuda(
    ), torch.from_numpy(weight).cuda()
  elif (args.sparsity_type == "column"):
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    column_l2_norm = LA.norm(weight2d, 2, axis=0)
    percentile = np.percentile(column_l2_norm, percent)
    under_threshold = column_l2_norm < percentile
    above_threshold = column_l2_norm > percentile
    weight2d[:, under_threshold] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape[0]):
      expand_above_threshold[:, i] = above_threshold[i]
    expand_above_threshold = expand_above_threshold.reshape(shape)
    weight = weight2d.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(
    ), torch.from_numpy(weight).cuda()
  elif (args.sparsity_type == "filter"):
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l2_norm = LA.norm(weight2d, 2, axis=1)
    percentile = np.percentile(row_l2_norm, percent)
    under_threshold = row_l2_norm < percentile
    above_threshold = row_l2_norm > percentile
    weight2d[under_threshold] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape[0]):
      expand_above_threshold[i, :] = above_threshold[i]
    weight = weight2d.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(
    ), torch.from_numpy(weight).cuda()
  else:
    raise SyntaxError("Unknown sparsity type")


def hard_prune(args, ADMM, model):
  """
  hard_pruning, or direct masking
  Args:
        model: contains weight tensors in cuda

  """
  print ("hard pruning")
  for (name, W) in model.named_parameters():
    if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
      continue
    _, cuda_pruned_weights = weight_pruning(
        args, W, ADMM.prune_ratios[name])  # get sparse model in cuda

    W.data = cuda_pruned_weights  # replace the data field in variable


def test_sparsity(args, model):
  """
  test sparsity for every involved layer and the overall compression rate

  """
  total_zeros = 0
  total_nonzeros = 0
  if args.sparsity_type == "irregular":
    for i, (name, W) in enumerate(model.named_parameters()):
      if 'bias' in name:
        continue
      W = W.cpu().detach().numpy()
      zeros = np.sum(W == 0)
      total_zeros += zeros
      nonzeros = np.sum(W != 0)
      total_nonzeros += nonzeros
      print ("sparsity at layer {} is {}".format(
          name, zeros / (zeros + nonzeros)))
    total_weight_number = total_zeros + total_nonzeros
    print (
        'overal compression rate is {}'.format(
            total_weight_number /
            total_nonzeros))
  elif args.sparsity_type == "column":
    for i, (name, W) in enumerate(model.named_parameters()):

      if 'bias' in name or name not in ADMM.prune_ratios:
        continue
      W = W.cpu().detach().numpy()
      shape = W.shape
      W2d = W.reshape(shape[0], -1)
      column_l2_norm = LA.norm(W2d, 2, axis=0)
      zero_column = np.sum(column_l2_norm == 0)
      nonzero_column = np.sum(column_l2_norm != 0)
      total_zeros += np.sum(W == 0)
      total_nonzeros += np.sum(W != 0)
      print ("column sparsity of layer {} is {}".format(
          name, zero_column / (zero_column + nonzero_column)))
    print ('only consider conv layers, compression rate is {}'.format(
        (total_zeros + total_nonzeros) / total_nonzeros))
  elif args.sparsity_type == "filter":
    for i, (name, W) in enumerate(model.named_parameters()):
      if 'bias' in name or name not in ADMM.prune_ratios:
        continue
      W = W.cpu().detach().numpy()
      shape = W.shape
      W2d = W.reshape(shape[0], -1)
      row_l2_norm = LA.norm(W2d, 2, axis=1)
      zero_row = np.sum(row_l2_norm == 0)
      nonzero_row = np.sum(row_l2_norm != 0)
      total_zeros += np.sum(W == 0)
      total_nonzeros += np.sum(W != 0)
      print ("filter sparsity of layer {} is {}".format(
          name, zero_row / (zero_row + nonzero_row)))
    print ('only consider conv layers, compression rate is {}'.format(
        (total_zeros + total_nonzeros) / total_nonzeros))


def admm_initialization(args, ADMM, model):
  if not args.admm:
    return
  for i, (name, W) in enumerate(model.named_parameters()):
    if name in ADMM.prune_ratios:
      # Z(k+1) = W(k+1)+U(k)  U(k) is zeros here
      _, updated_Z = weight_pruning(args, W, ADMM.prune_ratios[name])
      ADMM.ADMM_Z[name] = updated_Z


def z_u_update(
        args,
        ADMM,
        model,
        device,
        train_loader,
        optimizer,
        epoch,
        data,
        batch_idx):
  if not args.admm:
    return

  if epoch != 1 and (epoch - 1) % args.admm_epoch == 0 and batch_idx == 0:
    for i, (name, W) in enumerate(model.named_parameters()):
      if name not in ADMM.prune_ratios:
        continue
      Z_prev = None
      if (args.verbose):
        Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
      ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

      # equivalent to Euclidean Projection
      _, updated_Z = weight_pruning(
          args, ADMM.ADMM_Z[name], ADMM.prune_ratios[name])
      ADMM.ADMM_Z[name] = updated_Z
      if (args.verbose):
        print ("at layer {}. W(k+1)-Z(k+1): {}".format(name,
                                                        torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name])**2)).item()))
        print ("at layer {}, Z(k+1)-Z(k): {}".format(name,
                                                      torch.sqrt(torch.sum((ADMM.ADMM_Z[name] - Z_prev)**2)).item()))
      ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + \
          ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def append_admm_loss(args, ADMM, model, ce_loss, var_reg):
  '''
  append admm loss to cross_entropy loss
  Args:
      args: configuration parameters
      model: instance to the model class
      ce_loss: the cross entropy loss
      var_reg: variance regularization from intermediate features
  Returns:
      ce_loss(tensor scalar): original cross enropy loss
      admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
      ret_loss(scalar): the mixed overall loss
  '''
  admm_loss = {}

  if args.admm:

    for i, (name, W) in enumerate(
          model.named_parameters()):  # initialize Z (for both weights and bias)
      if name not in ADMM.prune_ratios:
        continue

      admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(
          W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2)**2) / (np.sqrt(W.numel()))
  
  mixed_loss = 0
  mixed_loss += ce_loss
  for k, v in admm_loss.items():
    mixed_loss += v
  mixed_loss += -1. * args.var_reg_weight * var_reg

  ret_dict = {
    "ce_loss": ce_loss,
    "admm_loss": admm_loss,
    "mixed_loss": mixed_loss,
    "variance_reg": var_reg
  }
  return ret_dict


def masked_retrain(args, model, device, train_loader, optimizer, epoch):
  if not args.masked_retrain:
    return

  model.train()
  masks = {}
  for i, (name, W) in enumerate(model.named_parameters()):
    if name not in ADMM.ADMM_Z:
      continue
    above_threshold, W = weight_pruning(args, W, ADMM.prune_ratios[name])
    masks[name] = above_threshold

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)

    loss.backward()

    for i, (name, W) in enumerate(model.named_parameters()):
      if name in masks:
        W.grad *= masks[name]

    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print ("cross_entropy loss: {}".format(loss))
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
  test_sparsity(args, model)


def train(args, model, device, train_loader, optimizer, epoch):
  model.train()

  admm_initialization(args, model)  # intialize Z variable

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    ce_loss = F.cross_entropy(output, target)

    z_u_update(args, model, device, train_loader, optimizer,
                epoch, data, batch_idx)   # update Z and U variables
    ret_dict = append_admm_loss(
        args, model, ce_loss)  # append admm losss

    ret_dict["mixed_loss"].backward()

    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print (
          "cross_entropy loss: {}, mixed_loss : {}".format(
              ret_dict["ce_loss"], ret_dict["mixed_loss"]))
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), ret_dict["ce_loss"].item()))
  if args.verbose:
    for k, v in ret_dict["admm_loss"].items():
      print ("at layer {}, admm loss is {}".format(k, v))
