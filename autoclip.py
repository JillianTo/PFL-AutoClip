##
## Adapted from https://github.com/pseeth/autoclip/blob/master/autoclip.py
## Pytorch Ignite dependency removed
##

import numpy as np
import torch

class AutoClip():
    def __init__(self, clip_percentile):
        self.grad_history = []
        self.clip_percentile=clip_percentile

    def _get_grad_norm(model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 

    def autoclip_gradient(self, model):
        obs_grad_norm = AutoClip._get_grad_norm(model)
        self.grad_history.append(obs_grad_norm)
        clip_value = np.percentile(self.grad_history, self.clip_percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
