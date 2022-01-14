""" Script for fast transferable attack. """
import copy
import time
import torch

from fta.attacks.base import BaseAttack

import pdb


class FastTransferableAttack(BaseAttack):
  """ FTA class with Pytorch ends. """
  def __init__(self,
               victim_model,
               distance,
               epsilon: float=16./255,
               step_size: float=2./255,
               num_steps: int=100,
               verbose: bool=False):
    super(FastTransferableAttack, self).__init__(
        "fta",
        distance=distance)
    self._verbose = verbose
    self._epsilon = epsilon
    self._step_size = step_size
    self._num_steps = num_steps
    self._victim_model = copy.deepcopy(victim_model)
    for p in self._victim_model.surrogate_model.parameters():
      p.requires_grad = False
    self._victim_model.surrogate_model.eval()
    self._loss_fn = torch.nn.MSELoss().cuda()

  def _attack_batch(self, input_batch):
    X_var = input_batch.clone().detach()
    benign_features = None
    if self._verbose:
      time_start = time.time()
    for idx in range(self._num_steps):
      X_var = X_var.clone().detach().requires_grad_(True)
      ret_dict = self._victim_model(X_var, batch_process=True)
      features = ret_dict["intermediate_feature"]
      if idx == 0:
        benign_features = \
            features.clone().detach().requires_grad_(False)
        loss_batch = -1. * torch.var(
            features.view(X_var.shape[0], -1),
            dim=1)
        loss = loss_batch.mean()
      else:
        loss = self._loss_fn(features, benign_features)
        # # Lipschitz constant loss
        # for batch_count in range(benign_features.shape[0]):
        #   for channel_count in range(benign_features.shape[1]):
        #     if batch_count == 0 and channel_count == 0:
        #       loss = torch.norm(benign_features[batch_count][channel_count] - features[batch_count][channel_count])
        #     else:
        #       loss += torch.norm(benign_features[batch_count][channel_count] - features[batch_count][channel_count])
        loss /= benign_features.shape[0] * benign_features.shape[1]
      
      loss.backward()
      grad = X_var.grad.data
      X_var = X_var.detach() + self._step_size * grad.sign_()

      if self._distance.name == "linf":
        X_var = torch.max(
            torch.min(
                X_var,
                input_batch + self._epsilon),
            input_batch - self._epsilon)
        X_var = torch.clamp(X_var, 0., 1.)
      else:
        raise NotImplementedError
    if self._verbose:
      time_end = time.time()
      print("Latency: {}".format(time_end - time_start))
    return X_var.detach()

  def _attack(self, input):
    return self._attack_batch(torch.unsqueeze(input, 0))[0]


FTA = FastTransferableAttack
