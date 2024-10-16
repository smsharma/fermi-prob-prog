import os
import sys
import dill as pickle
import argparse
import json

import numpy as np
import healpy as hp

import jax.numpy as jnp

wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
sys.path.append(f"{wdir}/..")
from models.np_model_gc import NPModelGC11, NPModelGC17, NPModelGC2, NPModelGC2SCF, NPModelGC7, NPModelGCFull
from models.np_model import NPModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-n', type=int, default=10000)
    parser.add_argument('--n_step', type=int, default=2000)
    parser.add_argument('--fit_type', type=str, default='test')
    parser.add_argument('--psf', type=str, default='king')
    parser.add_argument('--model', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--label', type=str)
    args = parser.parse_args()

    #===== DIRS =====#
    wdir = "/n/home07/yitians/fermi/fermi-prob-prog/production"
    data_dir = f"{wdir}/../outputs/simulations"
    save_dir = f"/n/holylabs/LABS/iaifi_lab/Users/yitians/fermi/fermi-prob-prog/outputs/fit/{args.fit_type}_{args.model}_{args.data}_{args.psf}psf_{args.label}"
    os.makedirs(save_dir, exist_ok=True)

    #===== MASK & DATA =====#
    mask_roi = jnp.load(f"{wdir}/mask_roi.npy")
    mask_norm = jnp.load(f"{wdir}/mask_norm.npy")

    data = np.load(f"{data_dir}/sim_{args.data}_{args.psf}psf_n100.npy")[args.i]
    data_full = np.zeros(hp.nside2npix(128))
    data_full[~mask_norm] = data
    data_in = jnp.array(data_full, dtype=jnp.int32)

    #===== PSF & MODEL =====#
    if args.psf == 'king':
        psf_tags = ['king', 'old']
    # elif args.psf == 'kingold':
    #     psf_tags = ['king', 'old']
    elif args.psf == 'delta':
        psf_tags = ['deltasimple']
        # print('USING OLD DELTA PSF FOR DEBUGING')
        # psf_tags = ['delta']
    else:
        raise NotImplementedError(args.psf)
    
    ModelDict = {
        'gc11': NPModelGC11,
        'gc17': NPModelGC17,
        'gc2' : NPModelGC2,
        'gc2scf' : NPModelGC2SCF,
        'gc7' : NPModelGC7,
        'gcfull': NPModelGCFull,
        'gcfullAlm': NPModelGCFull,
        'old': NPModel,
    }
    kwargs = dict(
        psf_tags=psf_tags,
        data=data_in,
    )
    if args.model == 'gcfullAlm':
        kwargs.update(dict(Alm=True))
    m = ModelDict[args.model](**kwargs)
    if args.model in ['gc2', 'gc2scf', 'gc7', 'gcfull', 'gcfullAlm']: # the models that requires truth_dict
        truth_dict = json.load(open(f"{wdir}/truth_dict_{args.data}.json"))
        m.set_truth(truth_dict)

    #===== FIT =====#
    if args.fit_type == 'svi':
        m.fit_svi(n_steps=args.n_step, data=data_in)
        samples = m.get_svi_samples(num_samples=args.n)

    elif args.fit_type in ['hmc', 'hmcnt']:
        if args.fit_type == 'hmcnt':
            m.fit_svi(n_steps=args.n_step, data=data_in)
        mcmc = m.run_nuts(
            num_chains=4, num_warmup=500, num_samples=args.n//4,
            use_neutra=(args.fit_type=='hmcnt'), data=data_in
        )
        samples = mcmc.get_samples()

    elif args.fit_type == 'test':
        mcmc = m.run_nuts(
            num_chains=4, num_warmup=10, num_samples=10,
            use_neutra=False, data=data_in
        )
        samples = mcmc.get_samples()

    else:
        raise NotImplementedError(args.fit_type)
    
    #===== SAVE =====#
    pickle.dump(samples, open(f"{save_dir}/{args.fit_type}_samples_i{args.i}_n{args.n}_ns{args.n_step}.p", 'wb'))
