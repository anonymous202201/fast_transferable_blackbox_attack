""" Script for generating AE examples.
"""

import argparse
import importlib
import os
import sys
from tqdm import tqdm

from fta.utils.distances import LINF
from fta.utils.torch_utils import image_utils

import pdb


# Example:
# python tools/generate.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --dataset_type $DATASET_TYPE --step_size $STEP_SIZE --num_steps $NUM_STEPS --attack_method $ATTACK_METHOD --surrogate_model $SURROGATE_MODEL --custimized_pretrain $CUSTIMIZED_PRETRAIN


def generate(args):
  if os.path.exists(args.output_dir):
    raise ValueError("Directory \"{}\" already exists.".format(args.output_dir))
  os.mkdir(args.output_dir)

  surrogate_lib = importlib.import_module(
      "fta.models.surrogates." + args.surrogate_model.split("_")[0])
  surrogate_cls = getattr(
      surrogate_lib, args.surrogate_model.replace("_", "").upper())
  surrogate = surrogate_cls(
      is_normalize=True,
      custimized_pretrain=args.custimized_pretrain)

  if args.dataset_type == "imagenet":
    img_size = (224, 224)
  elif args.dataset_type == 'bdd100k':
    img_size = (720, 1280)
  else:
    raise ValueError("Invalid dataset type: {}".format(args.dataset_type))
  print("Generated image size: {}".format(img_size))

  distance = LINF(budget=args.epsilon)

  attack_lib = importlib.import_module(
      "fta.attacks." + args.attack_method)
  attack_cls = getattr(attack_lib, args.attack_method.upper())
  print("Attack Parameters:")
  print("Attack method: {}".format(args.attack_method))
  print("epsilon: {}".format(args.epsilon))
  print("step_size: {}".format(args.step_size))
  print("num_steps: {}".format(args.num_steps))
  print("custimized_pretrain: {}".format(args.custimized_pretrain))
  attack = attack_cls(surrogate, distance,
                      epsilon=args.epsilon / 255.,
                      step_size=args.step_size / 255.,
                      num_steps=args.num_steps,
                      verbose=args.verbose)


  img_names = os.listdir(args.input_dir)
  for img_name in tqdm(img_names):
    img_name_noext = os.path.splitext(img_name)[0]

    img_path_benign = os.path.join(args.input_dir, img_name)
    img_benign_var = image_utils.load_img(
        img_path_benign,
        img_size = img_size,
        expand_batch_dim=True).cuda()
    
    img_adv_var = attack(img_benign_var, batch_process=True)
    output_img = image_utils.save_img(
        img_adv_var,
        os.path.join(args.output_dir, img_name_noext + "_adv.png"),
        with_batch_dim=True)
  return


def parse_args(args):
  parser = argparse.ArgumentParser(
      description="PyTorch AE generator.")
  parser.add_argument(
      '--input_dir',
      default="./sample_images",
      type=str)
  parser.add_argument(
      '--output_dir', default="./temp_outputs", type=str)
  parser.add_argument('--attack_method', default="fta", type=str)
  parser.add_argument(
      '--surrogate_model', default="vgg16_t", type=str)
  parser.add_argument(
      '--dataset_type', choices=['imagenet', 'bdd100k'])
  parser.add_argument(
      '--custimized_pretrain', type=str, default=None,
      help='for admm trained weights')
  parser.add_argument(
      '--step_size', default=1.0, type=float)
  parser.add_argument(
      '--epsilon', default=16.0, type=float)
  parser.add_argument(
      '--num_steps', default=20, type=int)
  parser.add_argument(
      '--verbose', action='store_true', default=False)
  return parser.parse_args(args)


def main(args=None):
  # parse arguments
  if args is None:
    args = sys.argv[1:]
  args = parse_args(args)
  args_dic = vars(args)

  generate(args)

if __name__ == "__main__":
  main()