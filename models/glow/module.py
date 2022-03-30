import torch
import pytorch_lightning as pl

from models.glow import Glow
from models.utils import NLLLoss, bits_per_dim

class GlowPL(pl.LightningModule):

    def __init__(self, num_channels=256, num_levels=5, num_steps=32, quants=256, lr=1e-4, optimizer_kwargs={'weight_decay': 1e-5}, scheduler='plateau', scheduler_kwargs={'patience':4}):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
        """
        super().__init__()
        self.flow = Glow(num_channels, num_levels, num_steps, quants=quants)
        self.loss = NLLLoss(k=quants)
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **optimizer_kwargs)

        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
        else:
            raise NotImplementedError

    def log_prob(self, x):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        
        z, sldj = self.flow(x)
        nll = self.loss(z, sldj)
        bpd = bits_per_dim(x, nll)
        
        return bpd

    def configure_optimizers(self):
        
        return {"optimizer": self.optimizer, 
                    "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1}
                }
    def training_step(self, batch, batch_idx):
        loss = self.log_prob(batch[0])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.log_prob(batch[0])
        self.log('val_loss', loss)