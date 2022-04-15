import logging
import argparse
from pathlib import Path

import numpy as np

from models.counterfactuals import GenerateCounterfactuals
from models.glow_module import GlowPL
from models.resnet_module import ResnetPL
from utils import create_mask as cm
from utils.cart import make_wcs, to_cart

import torch

def generate(data_dir, sample_name='data_uniform_test', idx_test=0):

    extent = 25
    n_pixels = 100
    pixelsize = 2 * extent / n_pixels
    nside = 128

    mask_ps = np.load("data/fermi_data/fermidata_pscmask.npy") == 1
    mask_roi = to_cart(cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2., mask_ring=True, inner=0, outer=60., custom_mask=mask_ps), n_pixels=n_pixels, pixelsize=pixelsize) > 0

    model_inference = ResnetPL(mask=mask_roi[2:-2, 2:-2], device='cpu')
    model_gen = GlowPL(num_channels=256, num_levels=5, num_steps=18, quants=2971., add_unif_noise=False)

    model_gen.flow.load_state_dict(torch.load("data/logs/wandb/latest-run/files/flow.ckpt", map_location=torch.device('cpu')))
    model_gen.flow.eval()  

    model_inference.resnet.load_state_dict(torch.load("data/resnet.pt", map_location=torch.device('cpu')))
    model_inference.eval()

    data = np.load("{}/samples/{}.npz".format(data_dir, sample_name))

    signal_ensemble = data["signal_ensemble"]

    x_test = torch.Tensor(signal_ensemble).unsqueeze(1)
    x_test = x_test[:, :, 2:-2, 2:-2] 

    x_cf = np.zeros((8, 96, 96))

    for i_comp in range(8):
        if i_comp == 5:
            continue
        print("Computing counterfactual for {}".format(i_comp))
        x0 = x_test[idx_test].unsqueeze(0)
        gc = GenerateCounterfactuals(x0, model_gen=model_gen, model_inf=model_inference, mask=mask_roi[2:-2, 2:-2], device='cpu', sigma=1.)
        if gc.mu[0, i_comp] < 0.01:
            continue
        x_cf[i_comp], z_cf = gc.generate_counterfactual(i_param=i_comp, cf_type=-1, lr=5e-1, print_every=10)

    np.save("data/counterfactuals/x_cf_{}.npy".format(idx_test), x_cf)

def parse_args():
    parser = argparse.ArgumentParser(description="Script to train conditional density estimator")

    # Command line arguments
    parser.add_argument("--sample", type=str, default='data_uniform_test', help='Sample name')
    parser.add_argument("--dir", type=str, default=".", help="Directory; training data will be loaded from the data/samples subfolder, model saved in the data/models subfolder")

    parser.add_argument("--idx_test", type=int, default=0, help='Index of test sample')

    # Training option
    return parser.parse_args()

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO,
    )
    logging.info("Hi!")

    args = parse_args()

    generate(data_dir="{}/data/".format(args.dir), sample_name=args.sample, idx_test=args.idx_test)

    logging.info("All done!")
