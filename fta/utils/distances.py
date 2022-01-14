"""Provides classes to measure the distance between two images."""

from abc import ABC
from abc import abstractmethod


class Distance(ABC):
  """Base class for distances
  This class should be subclassed when implementing
  new distances. Subclasses must implement _calculate.
  """

  def __init__(self,
               name,
               budget=None):
    self._name = name
    self._budget = budget

  @property
  def budget(self):
    return self._budget

  @property
  def name(self):
    return self._name

  @abstractmethod
  def calculate(self, img1, img2, **kwargs):
    raise NotImplementedError

  def is_inbound(self, img1, img2, **kwargs):
    return self.calculate(img1, img2, **kwargs) <= self._budget


class NormInfinityDistance(Distance):
  """Calculates L-infinity distance between images."""

  def __init__(self, budget=None):
    super(NormInfinityDistance, self).__init__("linf", budget)

  def calculate(self, img1, img2):
    """
    Returns Linf distance between images.

    Parameters
    ----------
    img1, img2 : ndarray or torch.tensor
      Image matrices to be calculated.

    Outputs
    -------
    L-infinity distance : float

    """
    assert img1.shape == img2.shape
    return float((img1 - img2).abs().max())


LINF = NormInfinityDistance
