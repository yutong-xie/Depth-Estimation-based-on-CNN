import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
<<<<<<< Updated upstream
=======

class Train_Loss(nn.Module):
  def __init__(self, param):
    super(Train_Loss, self).__init__()
    self.param = param

  class Train_Loss(nn.Module):
    def __init__(self, param = 0.5):
        super(Train_Loss, self).__init__()
        self.param = param

    def forward(self, pred, target):
        # mask out zero values and invalid regions
        mask = target > 0
        diff = pred[mask] - torch.log(target[mask])

        # the lambda parameter is set to 0.5
        loss = torch.mean(diff**2) - self.param * torch.pow(torch.mean(diff), 2)
        return loss
>>>>>>> Stashed changes
