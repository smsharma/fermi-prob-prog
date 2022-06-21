import numpy as np
import torch
import pytorch_lightning as pl

from models.glow import Glow

class GlowPL(pl.LightningModule):

    def __init__(self, mask=None, num_channels=256, num_levels=5, num_steps=18, quants=256, add_unif_noise=False, lr=1e-4, optimizer_kwargs={'weight_decay': 1e-5}, scheduler='plateau', scheduler_kwargs={'patience':4}):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
        """
        super().__init__()

        self.flow = Glow(in_channel=1, n_flow=num_steps, n_block=num_levels, filter_size=num_channels, quants=quants, add_unif_noise=add_unif_noise)
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.quants = quants
        self.mask = mask
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **optimizer_kwargs)

        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
        else:
            raise NotImplementedError

    def loss(self, x):
        """
        Given a batch of images, return the loss in bits per dimension (scaled negative log likelihood)
        """

        # Add dummy channel dimension and bring (h w) dim down to (96 96) for compatibility with squeeze/split
        x = x.unsqueeze(1)[:, :, 2:-2, 2:-2] 

        if self.mask != None:
            mask = self.mask.type_as(x).int()
            x = x * mask

        dim = np.prod(x.size()[1:])
        log_p_sum, logdet, _ = self.flow(x)
        log_prob = logdet + log_p_sum  # Note: log_p_sum already includes sldj from quantization step
        loss = -log_prob / (np.log(2) * dim)
        
        return loss.mean()

    def configure_optimizers(self):
        
        return {"optimizer": self.optimizer, 
                    "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1}
                }

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch[0])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch[0])
        self.log('val_loss', loss)