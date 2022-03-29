import sys
import json
import logging
import argparse

import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import WandbLogger

import mlflow

from models.glow import Glow
from models.glow.module import GlowPL

from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler

def train(data_dir, experiment_name, sample_name='data_uniform'):

    # Cache hyperparameters to log
    params_to_log = locals()

    logging.info("")
    logging.info("Creating estimator")
    logging.info("")

    # # MLFlow logger
    # tracking_uri = "file:{}/logs/mlruns".format(data_dir)
    # mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    # mlf_logger.log_hyperparams(params_to_log)

    pl.seed_everything(1)

    wandb_logger = WandbLogger(save_dir="{}/logs/".format(data_dir), name=experiment_name, project="fermi_counterfactuals", log_model=True)
    wandb_logger.log_hyperparams(params_to_log)

    data = np.load("{}/samples/{}.npz".format(data_dir, sample_name))

    signal_ensemble = data["signal_ensemble"]
    flux_fraction = data["flux_fraction"]

    x = torch.Tensor(signal_ensemble).unsqueeze(1)
    y = torch.Tensor(flux_fraction)

    x = x[:, :, 2:-2, 2:-2] 

    val_fraction = 0.10
    n_samples_val = int(val_fraction * len(x))

    dataset = TensorDataset(x, y)

    dataset_train, dataset_val = random_split(dataset, [len(x) - n_samples_val, n_samples_val])
    train_loader = DataLoader(dataset_train, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=128, num_workers=4, pin_memory=True, shuffle=False)

    model = GlowPL(num_channels=256, num_levels=5, num_steps=18, quants=x.max() + 1)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(max_epochs=2, gpus=4, strategy="ddp_find_unused_parameters_false", gradient_clip_val=1., callbacks=[checkpoint_callback, lr_monitor], logger=wandb_logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model.load_from_checkpoint(checkpoint_callback.best_model_path, num_channels=256, num_levels=5, num_steps=18, quants=x.max() + 1)

    # trainer.logger.experiment.log({"flow": model.flow})


    # if model.trainer.is_global_zero:

    #     # Save density estimator
    #     mlflow.set_tracking_uri(tracking_uri)
    #     with mlflow.start_run(run_id=mlf_logger.run_id):
    #         mlflow.pytorch.log_model(model.flow, "flow")    

    #     # Check to make sure model can be succesfully loaded
    #     model_uri = "runs:/{}/flow".format(mlf_logger.run_id)
    #     mlflow.pytorch.load_model(model_uri)

def parse_args():
    parser = argparse.ArgumentParser(description="Script to train conditional density estimator")

    # Command line arguments
    parser.add_argument("--sample", type=str, default='data_uniform', help='Sample name')
    parser.add_argument("--dir", type=str, default=".", help="Directory; training data will be loaded from the data/samples subfolder, model saved in the data/models subfolder")
    parser.add_argument("--name", type=str, default='test', help='Name used to store experiment')

    # Training option
    return parser.parse_args()

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO,
    )
    logging.info("Hi!")

    args = parse_args()

    train(data_dir="{}/data/".format(args.dir), experiment_name=args.name, sample_name=args.sample)

    logging.info("All done!")
