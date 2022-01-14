""" Utilities for admm based pruning with variance regularization.
"""
import torch
import yaml

import pdb


def get_pruning_names(admm_conf_path):
  with open(admm_conf_path, "r") as stream:
    try:
      raw_dict = yaml.load(stream)
      prune_ratios = raw_dict['prune_ratios']
    except yaml.YAMLError as exc:
      print(exc)
  return tuple(prune_ratios)


def get_layer(model, name, arch):
  tags = name.split('.')
  layer = getattr(model, tags[0])[int(tags[1])]
  return layer


def calculate_var_reg(features):
  assert len(features) > 0
  for idx, feature in enumerate(features):
    curt_var = torch.var(
      feature.view(feature.shape[0], -1),
      dim=1)
    if idx == 0:
      var_reg = curt_var
    else:
      var_reg += curt_var
  return var_reg.mean()
