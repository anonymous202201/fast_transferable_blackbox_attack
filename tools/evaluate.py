""" Script for evaluating AE examples.
"""

import argparse
import importlib
import os
import shutil
import sys
import torch
import torchvision
from tqdm import tqdm

from fta.utils.dataset_utils import imagenet_utils
from fta.utils.torch_utils import image_utils, model_utils

import pdb


# Sample Usage:
# CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py


def evaluate(args):
  imagenet_label_dict = imagenet_utils.load_imagenet_label_dict()
  target_model_type = args.target_model
  model_class = getattr(torchvision.models, args.target_model)
  model = model_class(pretrained=True).cuda()
  model.eval()

  img_mean, img_std = imagenet_utils.get_imagenet_normalize()
  torch_normalize = model_utils.Normalize(img_mean, img_std)

  img_names = os.listdir(args.benign_dir)
  acc_count = 0
  total_count = 0
  for img_name in tqdm(img_names):
    img_name_noext = os.path.splitext(img_name)[0]

    img_path_benign = os.path.join(args.benign_dir, img_name)
    img_benign_var = image_utils.load_img(
        img_path_benign, expand_batch_dim=True).cuda()
    img_benign_var = torch_normalize(img_benign_var)
    pred_benign = torch.argmax(model(img_benign_var), axis=1)
    pred_benign_id = pred_benign.cpu().numpy()[0]
    
    img_path_adv = os.path.join(
        args.adv_dir,
        img_name_noext + "_adv.png")
    if not os.path.exists(img_path_adv):
      print("adv image not found.")
      continue
    img_adv_var = image_utils.load_img(
        img_path_adv, expand_batch_dim=True).cuda()
    img_adv_var = torch_normalize(img_adv_var)
    pred_adv = torch.argmax(model(img_adv_var), axis=1)
    pred_adv_id = pred_adv.cpu().numpy()[0]
    
    print("ID: {0}, ori: {1}, adv: {2}".format(
        img_name_noext,
        imagenet_label_dict[pred_benign_id],
        imagenet_label_dict[pred_adv_id]))

    if pred_benign_id == pred_adv_id:
      acc_count += 1
    total_count += 1
  accuracy = float(acc_count) / float(total_count)
  print("Evaluate path: ", args.adv_dir)
  print("Target Model: ", args.target_model)
  print("ASR: ", 1.0 - accuracy)
  print("{} over {}".format(total_count - acc_count, total_count))
  return


def parse_args(args):
  parser = argparse.ArgumentParser(
      description="PyTorch AE evaluator.")
  parser.add_argument(
      '--benign_dir',
      default="./sample_images",
      type=str)
  parser.add_argument(
      '--adv_dir', default="./temp_outputs", type=str)
  parser.add_argument(
      '--target_model', default="resnet152", type=str)
  return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    evaluate(args)

if __name__ == "__main__":
   main()