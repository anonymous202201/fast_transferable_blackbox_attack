""" Adapted Resnet152 pytorch model that is used as surrogate. """
import torch
import torchvision

from fta.models.surrogates.base import BaseSurrogate
from fta.utils.dataset_utils.imagenet_utils import get_imagenet_normalize
from fta.utils.torch_utils.model_utils import Normalize

import pdb


class TruncatedResnet152PyTorch(torch.nn.Module):
  """ Truncated Resnet152 model. The model is cut in middle and
      output intermediate feature maps.
  """
  def __init__(self, layer_idx: int, is_normalize: bool,
               custimized_pretrain: str=""):
    super(TruncatedResnet152PyTorch, self).__init__()
    self._layer_idx = layer_idx
    self._is_normalize = is_normalize
    if self._is_normalize:
      img_mean, img_std = get_imagenet_normalize()
      self._normalize = Normalize(img_mean, img_std)
    base_model = torchvision.models.resnet152(
        pretrained=True).cuda().eval()
    if custimized_pretrain is not None:
      print("========== Loading custimized weights ==========")
      weight_dict = torch.load(custimized_pretrain)
      if isinstance(weight_dict, dict) and "state_dict" in weight_dict.keys():
        weight_dict = weight_dict["state_dict"]
      base_model.load_state_dict(weight_dict)
    features = list(base_model.children())[:self._layer_idx]
    self._features = torch.nn.ModuleList(features).cuda().eval()

  def forward(self, input_t):
    if self._is_normalize:
      x = self._normalize(input_t)
    else:
      x = input_t
    
    for curt_layer in self._features:
      x = curt_layer(x)
    return x


class TruncatedResnet152(BaseSurrogate):
  """ Truncated Resnet152 model wrapped in surrogate class.
  """
  def __init__(self, layer_idx: int=5, is_normalize: bool=True,
               custimized_pretrain: str=""):
    print("Layer idx: ", layer_idx)
    self._resnet152_truncated = TruncatedResnet152PyTorch(
        layer_idx=layer_idx, is_normalize=is_normalize,
        custimized_pretrain=custimized_pretrain)
    super(TruncatedResnet152, self).__init__(
        "resnet152truncated",
        surrogate_model=self._resnet152_truncated)

  def _predict(self, input):
    return self._predict_batch(torch.unsqueeze(input, 0))

  def _predict_batch(self, input_batch):
    ret = {}
    feature = self._resnet152_truncated(input_batch)
    ret["intermediate_feature"] = feature
    return ret


RESNET152T = TruncatedResnet152


class Resnet152(BaseSurrogate):
  """ Resnet152 model wrapped in surrogate class.
  """
  def __init__(self, is_normalize: bool=True, **kwargs):
    self._is_normalize = is_normalize
    if self._is_normalize:
      img_mean, img_std = get_imagenet_normalize()
      self._normalize = Normalize(img_mean, img_std)
    self._resnet152 = torchvision.models.resnet152(
        pretrained=True).cuda().eval()
    super(Resnet152, self).__init__(
        "resnet152", surrogate_model=self._resnet152)

  def _predict(self, input):
    return self._predict_batch(torch.unsqueeze(input, 0))

  def _predict_batch(self, input_batch):
    ret = {}
    logits = self._resnet152(input_batch)
    ret["logits"] = logits
    return ret


RESNET152 = Resnet152
