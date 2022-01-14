""" Adapted VGG16 pytorch model that is used as surrogate. """
import torch
import torchvision

from fta.models.surrogates.base import BaseSurrogate
from fta.utils.dataset_utils.imagenet_utils import get_imagenet_normalize
from fta.utils.torch_utils.model_utils import Normalize

import pdb


class TruncatedVgg16PyTorch(torch.nn.Module):
  """ Truncated VGG16 model. The model is cut in middle and output
  intermediate feature maps.
  """
  def __init__(self, layer_idx: int, is_normalize: bool,
               custimized_pretrain: str=""):
    super(TruncatedVgg16PyTorch, self).__init__()
    self._layer_idx = layer_idx
    self._is_normalize = is_normalize
    if self._is_normalize:
      img_mean, img_std = get_imagenet_normalize()
      self._normalize = Normalize(img_mean, img_std)
    base_model = torchvision.models.vgg16(
        pretrained=True).cuda().eval()
    if custimized_pretrain is not None:
      print("========== Loading custimized weights ==========")
      weight_dict = torch.load(custimized_pretrain)
      if isinstance(weight_dict, dict) and "state_dict" in weight_dict.keys():
        weight_dict = weight_dict["state_dict"]
      base_model.load_state_dict(weight_dict)
    features = list(base_model.features)[:self._layer_idx]
    self._features = torch.nn.ModuleList(features).cuda().eval()

  def forward(self, input_t):
    if self._is_normalize:
      x = self._normalize(input_t)
    else:
      x = input_t
    
    for curt_layer in self._features:
      x = curt_layer(x)
    return x


class TruncatedVgg16(BaseSurrogate):
  """ Truncated Vgg16 model wrapped in surrogate class.
  """
  def __init__(self, layer_idx: int=12, is_normalize: bool=True,
               custimized_pretrain: str=""):
    print("Layer idx: ", layer_idx)
    self._vgg16_truncated = TruncatedVgg16PyTorch(
        layer_idx=layer_idx, is_normalize=is_normalize,
        custimized_pretrain=custimized_pretrain)
    super(TruncatedVgg16, self).__init__(
        "vgg16truncated", surrogate_model=self._vgg16_truncated)

  def _predict(self, input):
    return self._predict_batch(torch.unsqueeze(input, 0))

  def _predict_batch(self, input_batch):
    ret = {}
    feature = self._vgg16_truncated(input_batch)
    ret["intermediate_feature"] = feature
    return ret


VGG16T = TruncatedVgg16


class Vgg16(BaseSurrogate):
  """ Vgg16 model wrapped in surrogate class.
  """
  def __init__(self, is_normalize: bool=True, **kwargs):
    self._is_normalize = is_normalize
    if self._is_normalize:
      img_mean, img_std = get_imagenet_normalize()
      self._normalize = Normalize(img_mean, img_std)
    self._vgg16 = torchvision.models.vgg16(
        pretrained=True).cuda().eval()
    super(Vgg16, self).__init__(
        "vgg16", surrogate_model=self._vgg16)

  def _predict(self, input):
    return self._predict_batch(torch.unsqueeze(input, 0))

  def _predict_batch(self, input_batch):
    ret = {}
    logits = self._vgg16(input_batch)
    ret["logits"] = logits
    return ret


VGG16 = Vgg16
