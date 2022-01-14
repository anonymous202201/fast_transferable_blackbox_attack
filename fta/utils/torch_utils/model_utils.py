import ast
import numpy as np
import torch
from torch.autograd import Variable


with open('fta/utils/coco91_label_dict.txt', 'r') as f:
  COCO91_LABEL_DICT = ast.literal_eval(f.read())


class Normalize(torch.nn.Module):
  def __init__(self, mean: tuple, std: tuple):
    super(Normalize, self).__init__()
    self._mean = mean
    self._std = std

  def forward(self, tensor: torch.tensor) -> torch.tensor:
    N, C, H, W = tensor.shape
    mean = np.expand_dims(np.expand_dims(np.expand_dims(
        np.array(self._mean).astype(np.float32),
        axis=0), axis=-1), axis=-1)
    mean_tile = torch.tensor(np.tile(mean, (N, 1, H, W)))
    std = np.expand_dims(np.expand_dims(np.expand_dims(
        np.array(self._std).astype(np.float32), axis=0),
        axis=-1), axis=-1)
    std_tile = torch.tensor(np.tile(std, (N, 1, H, W)))

    if tensor.is_cuda:
      mean_tile = mean_tile.cuda()
      std_tile = std_tile.cuda()

    tensor = (tensor - mean_tile) / std_tile
    return tensor


class UnNormalize(torch.nn.Module):
  def __init__(self, mean: tuple, std: tuple):
    super(UnNormalize, self).__init__()
    self.mean = mean
    self.std = std

  def forward(self, tensor: torch.tensor) -> torch.tensor:
    N, C, H, W = tensor.shape
    mean = np.expand_dims(np.expand_dims(np.expand_dims(
        np.array(self._mean).astype(np.float32), axis=0),
        axis=-1), axis=-1)
    mean_tile = torch.tensor(np.tile(mean, (N, 1, H, W)))
    std = np.expand_dims(np.expand_dims(np.expand_dims(
        np.array(self._std).astype(np.float32), axis=0),
        axis=-1), axis=-1)
    std_tile = torch.tensor(np.tile(std, (N, 1, H, W)))

    if tensor.is_cuda:
      mean_tile = mean_tile.cuda()
      std_tile = std_tile.cuda()

    tensor = tensor * std_tile + mean_tile
    return tensor


def numpy_to_variable(image, device=torch.device('cuda:0')):
  if len(image.shape) == 3:
    x_image = np.expand_dims(image, axis=0)
  else:
    x_image = image
  x_image = Variable(torch.tensor(x_image), requires_grad=True)
  x_image = x_image.to(device)
  x_image.retain_grad()
  return x_image

def variable_to_numpy(variable):
  return variable.cpu().detach().numpy()

def convert_torch_det_output(torch_out,
                             cs_th=0.5,
                             cls_as_name=False,
                             cls_filter=None):
  '''convert pytorch detection model output to list of dictionary of list
      [
          {
              'scores': [0.97943294], 
              'classes': [14], 
              'boxes': [[65.1657, 17.7265, 418.3291, 314.5997]]
          }
      ]
  '''
  ret_list = []
  for temp_torch_out in torch_out:
    temp_dic = {
        'scores' : [],
        'classes' : [],
        'boxes' : []
    }
    box_list = temp_torch_out['boxes'].cpu().numpy()
    score_list = temp_torch_out['scores'].cpu().numpy()
    label_list = temp_torch_out['labels'].cpu().numpy()
    for box, score, label in zip(box_list, score_list, label_list):
      if cls_filter != None and label not in cls_filter:
        continue
      if score < cs_th:
        continue
      temp_dic['scores'].append(score)
      temp_dic['boxes'].append(box)
      if cls_as_name:
        temp_dic['classes'].append(COCO91_LABEL_DICT[label])
      else:
        temp_dic['classes'].append(label)
    ret_list.append(temp_dic)
  return ret_list
