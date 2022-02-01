import numpy as np

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import pyro.distributions as dist

from models.layers import ConcatLayer, CustomSequential
from models.estimators import ema_loss

class MINE(nn.Module):
    def __init__(self, T, loss='mine', alpha=1., method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method
        
        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

        self.opt = torch.optim.Adam(self.T.parameters(), lr=1e-4)

    def forward(self, x, z, z_marg=None):

        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        if self.loss in ['mine', 'fdiv']:

            t = self.T(x, z).mean()
            t_marg = self.T(x, z_marg)

            if self.loss in ['mine']:
                second_term, self.running_mean = ema_loss(
                    t_marg, self.running_mean, self.alpha)
            elif self.loss in ['fdiv']:
                second_term = torch.exp(t_marg - 1).mean()

            return -t + second_term

        
    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi
    
    def optimize(self, x, z, iters=100, batch_size=64):
        
        mu_mi_list = []
        
        for iter in tqdm(range(iters)):
                                        
            self.opt.zero_grad()

            idxs = np.random.choice(x.shape[0], batch_size) 
            loss = self.forward(x[idxs,:], z[idxs,:])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            self.opt.step()

            mu_mi = -loss.item()

            mu_mi_list += [mu_mi / batch_size]

            # print(f"It {iter} - MI: {mu_mi / batch_size}")
                
        return mu_mi_list
