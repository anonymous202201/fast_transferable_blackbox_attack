""" Adapted mobilenet_v2 pytorch model that is used as surrogate. """
import torch
import torchvision

from fta.models.surrogates.base import BaseSurrogate
from fta.utils.dataset_utils.imagenet_utils import get_imagenet_normalize
from fta.utils.torch_utils.model_utils import Normalize

import pdb


class TruncatedMobilenetV2PyTorch(torch.nn.Module):
  """ Truncated mobilenet_v2 model. The model is cut in middle and output
  intermediate feature maps.
  """
  def __init__(self, layer_idx: int, is_normalize: bool,
               custimized_pretrain: str=""):
    super(TruncatedMobilenetV2PyTorch, self).__init__()
    self._layer_idx = layer_idx
    self._is_normalize = is_normalize
    if self._is_normalize:
      img_mean, img_std = get_imagenet_normalize()
      self._normalize = Normalize(img_mean, img_std)
    base_model = torchvision.models.mobilenet_v2(
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


class TruncatedMobilenet_V2(BaseSurrogate):
  """ Truncated mobilenet_v2 model wrapped in surrogate class.
  """
  def __init__(self, layer_idx: int=7, is_normalize: bool=True,
               custimized_pretrain: str=""):
    """ 
      mobile net classification best layer_idx: 7
      mobile net detection best layer_idx: 13
    """
    print("Layer idx: ", layer_idx)
    self._mobilenetv2_truncated = TruncatedMobilenetV2PyTorch(
        layer_idx=layer_idx, is_normalize=is_normalize,
        custimized_pretrain=custimized_pretrain)
    super(TruncatedMobilenet_V2, self).__init__(
        "mobilenetv2truncated", surrogate_model=self._mobilenetv2_truncated)

  def _predict(self, input):
    return self._predict_batch(torch.unsqueeze(input, 0))

  def _predict_batch(self, input_batch):
    ret = {}
    feature = self._mobilenetv2_truncated(input_batch)
    ret["intermediate_feature"] = feature
    return ret


MOBILENETV2T = TruncatedMobilenet_V2


class Mobilenet_V2(BaseSurrogate):
  """ mobilenet_v2 model wrapped in surrogate class.
  """
  def __init__(self, is_normalize: bool=True, **kwargs):
    self._is_normalize = is_normalize
    if self._is_normalize:
      img_mean, img_std = get_imagenet_normalize()
      self._normalize = Normalize(img_mean, img_std)
    self._mobilenetv2 = torchvision.models.mobilenet_v2(
        pretrained=True).cuda().eval()
    super(Mobilenet_V2, self).__init__(
        "mobilenetv2", surrogate_model=self._mobilenetv2)

  def _predict(self, input):
    return self._predict_batch(torch.unsqueeze(input, 0))

  def _predict_batch(self, input_batch):
    ret = {}
    logits = self._mobilenetv2(input_batch)
    ret["logits"] = logits
    return ret


MOBILENETV2 = Mobilenet_V2
