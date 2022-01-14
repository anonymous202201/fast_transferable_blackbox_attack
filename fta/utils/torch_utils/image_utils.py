import numpy as np
from PIL import Image
import torch
import torchvision

import pdb


def load_img(img_path: str,
             img_size: tuple=(224, 224),
             expand_batch_dim: bool=True,
             return_np: bool=False) -> torch.Tensor:
  if img_size == None:
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
  else:
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), 
        torchvision.transforms.ToTensor()])
  img_var = img_transforms(
      Image.open(img_path).convert('RGB'))
  if expand_batch_dim:
    img_var = torch.unsqueeze(img_var, dim=0)
  if return_np:
    return img_var.cpu().numpy()
  else:
    img_var = torch.autograd.Variable(img_var, requires_grad=True)
    img_var.retain_grad()
    return img_var


def save_img(img_var: torch.Tensor,
             save_path: str,
             with_batch_dim: bool=True) -> None:
  img_np = img_var.detach().cpu().numpy()
  if with_batch_dim:
    img_np = img_np[0]
  img_np = np.transpose(img_np, (1, 2, 0))
  Image.fromarray((img_np * 255.).astype(np.uint8)).save(save_path)
  return


def save_bbox_img(img, bbox_list, from_path=True, out_file='temp.jpg'):
  from PIL import Image, ImageDraw

  if from_path:
    source_img = Image.open(img).convert("RGB")
  else:
    source_img = Image.fromarray(img)

  draw = ImageDraw.Draw(source_img)
  for top, left, bottom, right in bbox_list:
    draw.rectangle([int(left), int(top), int(right), int(bottom)])

  source_img.save(out_file)
