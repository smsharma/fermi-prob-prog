import os
import sys
import json
import logging
import argparse
import glob
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

from models.glow_module import GlowPL
from utils.dataloader import BigDataset
from utils.cart import to_cart

def train(data_dir, experiment_name, sample_name='train', mask_type=None,
        batch_size=128,
        num_channels=256, num_levels=5, num_steps=18, quants=20000, add_unif_noise=True,
        max_epochs=80,
        gradient_clip_val=0.5,
        val_fraction=0.1,
        lr=3e-4,
        optimizer_kwargs={'weight_decay': 1e-5},
        scheduler='cosine',
        scheduler_kwargs=None,
        n_files=15,
        dataset_type="memmap"  # concat or memmap
        ):

    # Cache hyperparameters to log
    params_to_log = locals()

    scheduler_kwargs = {'T_max':max_epochs} if scheduler_kwargs is None else scheduler_kwargs

    logging.info("")
    logging.info("Creating estimator")
    logging.info("")

    # Make sure all GPUs have same seed
    pl.seed_everything(42)

    wandb_logger = WandbLogger(save_dir="{}/logs/".format(data_dir), group=experiment_name, name="run-{}".format(wandb.util.generate_id()), project="fermi_counterfactuals")
    wandb_logger.log_hyperparams(params_to_log)

    x_files = glob.glob("data/samples/x_{}_*".format(sample_name))[:n_files]
    theta_files = glob.glob("data/samples/theta_{}_*".format(sample_name))[:n_files]

    if mask_type == "roi":
        mask = (torch.Tensor(np.load("data/fermi_data/mask_roi.npy"))[2:-2, 2:-2] == 0.).bool().int()
    elif mask_type == "plane":
        mask = (torch.Tensor(np.load("data/fermi_data/mask_plane.npy"))[2:-2, 2:-2] == 0.).bool().int()
    else:
        mask = None

    if dataset_type == "concat":
    
        ## ConcatDataset dataset 
        datasets = [TensorDataset(torch.Tensor(np.load(x)), torch.Tensor(np.load(y))) for (x, y) in zip(x_files, theta_files)]
        dataset = ConcatDataset(datasets)
        n_samples_val = int(val_fraction * dataset.cumulative_sizes[-1])
        dataset_train, dataset_val = random_split(dataset, [dataset.cumulative_sizes[-1] - n_samples_val, n_samples_val])

    elif dataset_type == "memmap":

        ## memmap dataset
        dataset = BigDataset(x_files, theta_files)
        n_samples_val = int(val_fraction * dataset.data_count)
        dataset_train, dataset_val = random_split(dataset, [dataset.data_count - n_samples_val, n_samples_val])
    else:
        raise NotImplementedError

    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=False)

    model = GlowPL(mask=mask, num_channels=num_channels, num_levels=num_levels, num_steps=num_steps, quants=quants, add_unif_noise=add_unif_noise,
                lr=lr, scheduler=scheduler, optimizer_kwargs=optimizer_kwargs, scheduler_kwargs=scheduler_kwargs)

    wandb_logger.experiment

    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_DIR"] = wandb.run.dir

    checkpoint_path = "{}/checkpoints/".format(os.environ["WANDB_DIR"])

    path = Path(checkpoint_path)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=checkpoint_path, filename="{epoch:02d}-{val_loss:.2f}", every_n_epochs=1)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logging.info("Checkpoint path is {}".format(checkpoint_path))

    trainer = pl.Trainer(max_epochs=max_epochs, strategy="ddp_find_unused_parameters_false", accelerator="gpu", devices=-1, gradient_clip_val=gradient_clip_val, callbacks=[checkpoint_callback, lr_monitor], logger=wandb_logger)

    # trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=[0], gradient_clip_val=gradient_clip_val, callbacks=[checkpoint_callback, lr_monitor], logger=wandb_logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if model.trainer.is_global_zero:
        model.load_from_checkpoint(checkpoint_callback.best_model_path, num_channels=num_channels, num_levels=num_levels, num_steps=num_steps, quants=quants, add_unif_noise=add_unif_noise)
        torch.save(model.flow, "{}/flow.pt".format(wandb.run.dir))
        torch.save(model.flow.state_dict(), "{}/flow.ckpt".format(wandb.run.dir))

def parse_args():
    parser = argparse.ArgumentParser(description="Script to train conditional density estimator")

    # Command line arguments
    parser.add_argument("--sample", type=str, default='train', help='Sample name')
    parser.add_argument("--dir", type=str, default=".", help="Directory; training data will be loaded from the data/samples subfolder, model saved in the data/models subfolder")
    parser.add_argument("--name", type=str, default='production_run', help='Name used to store experiment')
    parser.add_argument("--add_unif_noise", type=int, default=1, help='Whether to add uniform noise during dequantization')
    parser.add_argument("--mask_type", type=str, default="None", help='"plane" or "roi" mask, optionally')

    # Training option
    return parser.parse_args()

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO,
    )
    logging.info("Hi!")

    args = parse_args()

    # If using a ROI mask, set maximum photons to a smaller value
    if args.mask_type == "roi":
        quants = 1000
    else:
        quants = 20000

    train(mask_type=args.mask_type, data_dir="{}/data/".format(args.dir), experiment_name=args.name, sample_name=args.sample, add_unif_noise=args.add_unif_noise, quants=quants)

    logging.info("All done!")
