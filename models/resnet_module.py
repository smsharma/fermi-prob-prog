
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytorch_lightning as pl

from models.resnet import ResNetEstimator

class ResnetPL(pl.LightningModule):

    def __init__(self, softmax=True, mask=None, lr=3e-4, optimizer_kwargs={'weight_decay': 1e-5}, scheduler='plateau', scheduler_kwargs={'patience':4}, device='cuda'):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
        """
        super().__init__()

        self.softmax = softmax

        self.resnet = ResNetEstimator(n_out=int(2 * 8))
        self.loss = nn.GaussianNLLLoss(reduction='mean')
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        
        self.mask = torch.from_numpy(mask).to(device)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **optimizer_kwargs)

        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        
        return {"optimizer": self.optimizer, 
                    "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1}
                }
    def training_step(self, batch, batch_idx):
        x, y = batch
        x += 1e-6
        x = torch.log10(x)
        out = self.resnet(x[:, 0] * ~self.mask)
        mu, logvar = torch.chunk(out, 2, -1)
        if self.softmax:
            mu = torch.softmax(mu, dim=-1)
        loss = self.loss(mu, y, logvar.exp())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x += 1e-6
        x = torch.log10(x)
        out = self.resnet(x[:, 0] * ~self.mask)
        mu, logvar = torch.chunk(out, 2, -1)
        if self.softmax:
            mu = torch.softmax(mu, dim=-1)
        loss = self.loss(mu, y, logvar.exp())
        self.log('val_loss', loss)