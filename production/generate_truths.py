import numpy as np
import json
import os
import re


def sample_from_prior(prior, rng):
    """Sample a truth dict from the prior specification.

    Args:
        prior (dict): Prior dict loaded from np_model_prior.json.
        rng (np.random.Generator): Numpy random generator.

    Returns:
        dict: A truth dict compatible with NPModel.simulate().
    """
    vd = {}
    for key, val in prior.items():
        if isinstance(val, str) and val.startswith("Dirichlet"):
            dim = int(re.search(r'\d+', val).group())
            vd[key] = rng.dirichlet(np.ones(dim)).tolist()
        elif isinstance(val, list) and len(val) == 2:
            vd[key] = rng.uniform(val[0], val[1])
        else:
            raise ValueError(f"Unknown prior format for '{key}': {val}")
    return vd


if __name__ == '__main__':

    truth_name = 'fullprior42'
    n_sim = 100
    seed = 42

    prior_path = os.path.join(os.path.dirname(__file__), '../src/fpp/models/np_model_prior.json')
    prior = json.load(open(prior_path, 'r'))

    child_rngs = np.random.default_rng(seed).spawn(n_sim)

    truths = [sample_from_prior(prior, rng=child_rngs[i]) for i in range(n_sim)]

    out_path = f"../outputs/truths/truths_{truth_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(truths, f, indent=2)
    print(f"Saved {n_sim} truths to {out_path}")
