"""Base class for surrogate models."""

from abc import ABC
from abc import abstractmethod


class BaseSurrogate(ABC):
  """
  Abstract base class for surrogate models.

  The :class:`BaseSurrogate` class represents a surrogate model for
  adversarial example generating method. It should be subclassed when
  implementing new surrogates.
  """

  def __init__(self,
               name,
               surrogate_model=None):
    self._name = name
    self._surrogate_model = surrogate_model

  @property
  def surrogate_model(self):
    return self._surrogate_model

  def __call__(self, input, batch_process: bool=True, **kwargs):
    """ Generates adversarial example(s) on single image or image
    batch.
    """
    assert isinstance(batch_process, bool)
    if batch_process:
      return self._predict_batch(input, **kwargs)
    else:
      return self._predict(input, **kwargs)
  
  @abstractmethod
  def _predict(self, input, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def _predict_batch(self, input_batch, **kwargs):
    raise NotImplementedError

