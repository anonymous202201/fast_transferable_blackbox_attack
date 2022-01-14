"""Base class for adversarial attacks."""

from abc import ABC
from abc import abstractmethod

from fta.utils.distances import LINF


class BaseAttack(ABC):
  """
  Abstract base class for adversarial attacks.

  The :class:`BaseAttack` class represents an adversarial example
  generating method that can mislead predictions of the model with
  limited perturbation. It should be subclassed when implementing new
  attacks.
  """

  def __init__(self,
               name,
               distance=LINF()):
    self._name = name
    self._distance = distance

    # to customize the initialization in subclasses, please
    # try to overwrite _initialize instead of __init__ if
    # possible
    self._initialize()

  def _initialize(self):
    """Additional initializer that can be overwritten by
    subclasses without redefining the full `__init__` method
    including all arguments and documentation.
    """
    pass

  def __call__(self, input, batch_process: bool=True, **kwargs):
    """ Generates adversarial example(s) on single image or image
    batch.
    """
    assert isinstance(batch_process, bool)
    if batch_process:
      return self._attack_batch(input, **kwargs)
    else:
      return self._attack(input, **kwargs)
  
  @abstractmethod
  def _attack(self, input, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def _attack_batch(self, input_batch, **kwargs):
    raise NotImplementedError
