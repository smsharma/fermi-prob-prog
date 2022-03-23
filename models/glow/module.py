import torch
import pytorch_lightning as pl

from models.glow import Glow
from models.utils import NLLLoss, bits_per_dim

class GlowPL(pl.LightningModule):

    def __init__(self, num_channels=256, num_levels=3, num_steps=16, quants=256):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
        """
        super().__init__()
        self.flow = Glow(num_channels, num_levels, num_steps, quants=quants)
        self.loss = NLLLoss(k=quants)

    def log_prob(self, x):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        
        z, sldj = self.flow(x)
        nll = self.loss(z, sldj)
        # bpd = bits_per_dim(x, nll)
        
        return nll  # bpd

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
        
        return {"optimizer": optimizer, 
                    "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1}
                }
    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self.log_prob(batch[0])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.log_prob(batch[0])
        self.log('val_loss', loss)